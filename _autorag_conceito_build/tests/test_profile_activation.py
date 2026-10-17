"""Profile registration can preserve the selected model without inference."""
import asyncio
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

import api
from autorag.core import Project, digest, write_json
from autorag.service import register_profile, sync_base


GEMMA = {"perfil_id": "pf_gemma_existing", "provedor": "ollama",
         "modelo": "embeddinggemma:300m-qat-q4_0", "dimensoes": 256,
         "revisao_modelo": "synthetic-weights-test-only", "normalizacao": "l2"}
MINILM = {"perfil_id": "pf_minilm_registered", "provedor": "sentence_transformers",
          "modelo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          "dimensoes": 384, "revisao_modelo": "synthetic-weights-test-only",
          "normalizacao": "l2"}


class ProfileActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Project(Path(self.temp.name)/"project")
        write_json(self.project.base_path, {
            "schema_version": "3.0", "documentos": [{
                "documento_id": "doc_one", "nome_documento": "Norma de teste",
                "estrutura": [{"topico_id": "topic_one", "titulo": "Regra",
                               "ativo": True, "resumo": "Resumo de teste.",
                               "conteudo_fonte": "Texto normativo de teste."}],
            }], "controle_embeddings": {"revisao_base": "base_test"},
        })
        self.project.save_config({"schema_version": "1.0", "perfis": {GEMMA["perfil_id"]: deepcopy(GEMMA)},
                                  "perfil_ativo_id": GEMMA["perfil_id"]})
        vector = np.zeros((1, 256)); vector[0, 0] = 1
        record = {"embedding_id": "topic_one::resumo::0000", "topico_id": "topic_one",
                  "documento_id": "doc_one", "categoria": "resumo", "segmento": 0,
                  "hash_texto_fonte": digest("Resumo de teste."), "texto_segmento": "Resumo de teste."}
        self.project.publish(GEMMA, "resumo", [record], vector, vector)
        sync_base(self.project)

    def tearDown(self):
        self.temp.cleanup()

    def old_index_files(self):
        return {str(path.relative_to(self.project.root)): path.read_bytes()
                for path in (self.project.root/"indices").rglob("*") if path.is_file()}

    def assert_existing_vectors_unchanged(self, original):
        self.assertEqual(self.old_index_files(), original)
        stored = self.project.load_index(GEMMA["perfil_id"], "resumo")
        self.assertEqual(stored[1].shape, (1, 256))
        categories = self.project.base()["documentos"][0]["estrutura"][0]["embeddings"]["perfis"][GEMMA["perfil_id"]]["categorias"]
        self.assertEqual(categories["resumo"]["status"], "atualizado")

    def test_register_without_activation_preserves_gemma_and_existing_indices(self):
        before = self.old_index_files()
        with patch("autorag.service.profile_spec", return_value=deepcopy(MINILM)) as spec:
            profile = register_profile(self.project, "sentence_transformers", MINILM["modelo"], 384, activate=False)
        self.assertEqual(profile, MINILM)
        self.assertEqual(self.project.config()["perfil_ativo_id"], GEMMA["perfil_id"])
        self.assertEqual(self.project.base()["controle_embeddings"]["perfil_ativo_id"], GEMMA["perfil_id"])
        self.assertIn(MINILM["perfil_id"], self.project.config()["perfis"])
        self.assertNotIn("activate", spec.call_args.kwargs)
        self.assert_existing_vectors_unchanged(before)

    def test_default_still_activates_new_profile_without_dropping_existing_indices(self):
        before = self.old_index_files()
        with patch("autorag.service.profile_spec", return_value=deepcopy(MINILM)):
            register_profile(self.project, "sentence_transformers", MINILM["modelo"], 384)
        self.assertEqual(self.project.config()["perfil_ativo_id"], MINILM["perfil_id"])
        self.assertEqual(self.project.base()["controle_embeddings"]["perfil_ativo_id"], MINILM["perfil_id"])
        self.assertIn(GEMMA["perfil_id"], self.project.config()["perfis"])
        self.assert_existing_vectors_unchanged(before)

    def test_false_does_not_select_a_first_profile_implicitly(self):
        empty = Project(Path(self.temp.name)/"empty")
        with patch("autorag.service.profile_spec", return_value=deepcopy(MINILM)):
            register_profile(empty, "sentence_transformers", MINILM["modelo"], 384, activate=False)
        self.assertIsNone(empty.config()["perfil_ativo_id"])
        self.assertIn(MINILM["perfil_id"], empty.config()["perfis"])

    def test_nonboolean_service_option_is_rejected_before_profile_lookup(self):
        with patch("autorag.service.profile_spec") as spec:
            with self.assertRaisesRegex(ValueError, "booleano"):
                register_profile(self.project, "sentence_transformers", MINILM["modelo"], 384, activate="false")
        spec.assert_not_called()

    def test_api_propagates_false_and_default_true(self):
        before = self.old_index_files()
        with patch.object(api, "project", self.project), \
             patch.dict(os.environ, {"AUTORAG_API_TOKEN": "activation-test-token", "AUTORAG_ENV": "development"}), \
             patch("autorag.service.profile_spec", return_value=deepcopy(MINILM)):
            with TestClient(api.app, headers={"Authorization": "Bearer activation-test-token"}) as client:
                body = {"provider": "sentence_transformers", "model": MINILM["modelo"], "dimensions": 384}
                response = client.post("/api/profiles", json={**body, "activate": False})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.project.config()["perfil_ativo_id"], GEMMA["perfil_id"])
                self.assertEqual(self.project.base()["controle_embeddings"]["perfil_ativo_id"], GEMMA["perfil_id"])
                self.assert_existing_vectors_unchanged(before)
                response = client.post("/api/profiles", json=body)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.project.config()["perfil_ativo_id"], MINILM["perfil_id"])

    def test_cloud_adapter_confirms_profile_checkpoint_before_http_success(self):
        from cloudflare.adapter import install
        events = []
        class Store:
            def checkpoint(self, project):
                events.append(("checkpoint", project.config()["perfil_ativo_id"]))
        async def request(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
                response = await client.post("/api/profiles", json={
                    "provider": "sentence_transformers", "model": MINILM["modelo"],
                    "dimensions": 384, "activate": False,
                }, headers={"Authorization": "Bearer activation-test-token"})
                events.append(("response", response.status_code))
                return response
        with patch.object(api, "project", self.project), patch.object(api, "submit", api.submit), \
             patch.dict(os.environ, {"AUTORAG_API_TOKEN": "activation-test-token"}), \
             patch("autorag.service.profile_spec", return_value=deepcopy(MINILM)):
            response = asyncio.run(request(install(api, Store())))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(events, [("checkpoint", GEMMA["perfil_id"]), ("response", 200)])
        self.assertEqual(self.project.config()["perfil_ativo_id"], GEMMA["perfil_id"])


if __name__ == "__main__":
    unittest.main()
