"""Prepared-base transfer regressions: no model calls or external network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from fastapi.testclient import TestClient

import api
from autorag.base_import import (canonical_digest, import_prepared_base,
                                 prepare_base_import)
from autorag.core import CATEGORIES, Project, digest, full_topic, read_json, write_json


def fixture():
    return {"schema_version": "2.1", "base_id": "normas_teste", "documentos": [{
        "documento_id": "norma_original", "nome_documento": "Portaria original",
        "ordem_documento": 1,
        "identificacao": {"tipo_documento": "Portaria", "identificacao_documento": "12"},
        "metadados": {"eixos": ["Administração"], "idioma": "português",
                      "custom": {"embeddings": "metadado do usuário, não controle"}},
        "vigencia": {"inicio_vigencia": "2025-01-01", "fim_vigencia": None},
        "estrutura": [{
            "topico_id": f"topico_{i}", "titulo": f"Artigo {i}", "ativo": i != 3,
            "ordem_no_documento": i, "caminho_hierarquico": ["Regras"],
            "conteudo_fonte": f"Art. {i} Regra pública número {i}.",
            "resumo": f"Resumo {i}.", "perguntas_sugeridas": [f"O que diz o artigo {i}?"],
            "resposta_sintese": f"O artigo {i} contém a regra.", "base_normativa": ["Portaria 12"],
            "palavras_chave": ["administração"], "fonte": {"pagina_inicio": i},
            "embedding": {"modelo": "obsoleto", "vetor": [999]},
            "embeddings": {"perfis": {"pf_antigo": {}}},
        } for i in (1, 2, 3)],
    }], "controle_embeddings": {"perfis": {"não_importar": {}}}}


def encoded(base):
    return json.dumps(base, ensure_ascii=False).encode("utf-8")


def legacy_npz(base, **changes):
    active = [t for d in base["documentos"] for t in d["estrutura"] if t.get("ativo", True)]
    values = {
        "vetores": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "topico_ids": np.asarray([t["topico_id"] for t in active]),
        "hashes_texto": np.asarray([digest(full_topic(t)) for t in active]),
        "modelo": np.asarray(["text-embedding-3-small"]),
        "dimensoes": np.asarray([3], dtype=np.int32),
        "normalizacao": np.asarray(["l2"]),
    }
    values.update(changes)
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


class BaseImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Project(Path(self.temp.name)/"project")

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self):
        return {str(p.relative_to(self.project.root)): p.read_bytes()
                for p in self.project.root.rglob("*") if p.is_file()}

    def test_complete_content_ids_metadata_and_inactive_flags_roundtrip(self):
        original = fixture()
        prepared = prepare_base_import(encoded(original))
        result = import_prepared_base(self.project, prepared)
        exported = self.project.base()
        self.assertEqual((result["documents"], result["topics"], result["active_topics"], result["inactive_topics"]), (1, 3, 2, 1))
        self.assertEqual(result["canonical_digest"], canonical_digest(original))
        self.assertEqual(result["base_canonical_digest"], canonical_digest(exported))
        self.assertEqual(canonical_digest(original), canonical_digest(exported))
        self.assertEqual(exported["base_id"], original["base_id"])
        self.assertEqual(exported["documentos"][0]["metadados"], original["documentos"][0]["metadados"])
        self.assertFalse(exported["documentos"][0]["estrutura"][2]["ativo"])
        self.assertTrue(all("embedding" not in t for t in exported["documentos"][0]["estrutura"]))
        self.assertNotIn("não_importar", self.project.config()["perfis"])

    def test_initial_import_preserves_root_order_taxonomy_and_historical_provenance(self):
        base = fixture()
        base["ordem_documentos"] = ["norma_original"]
        base["eixos"] = [{"eixo_id": "classe_original", "nome_eixo": "Administração", "documentos_ids": ["norma_original"]}]
        base["execucoes_processamento"] = [{"execucao_id": "run_original", "tipo": "topicos", "inicio": "2025-01-01",
            "fim": None, "status": "interrompido", "modelo": "modelo_anterior", "filtro_documentos_ids": ["norma_original"],
            "resultados": [{"origem": "C:/historico/norma.pdf"}], "resumo": {"processados": 1}}]
        base["estatisticas"] = {"documentos": 999, "topicos_ativos": 999}
        result = import_prepared_base(self.project, prepare_base_import(encoded(base)))
        exported = self.project.base()
        for key in ("base_id", "ordem_documentos", "eixos", "execucoes_processamento"):
            self.assertEqual(exported[key], base[key])
        self.assertTrue(result["root_metadata_preserved"])
        self.assertEqual(exported["estatisticas"]["documentos"], 1)
        self.assertEqual(exported["estatisticas"]["topicos_ativos"], 2)
        self.assertEqual(exported["estatisticas"]["topicos_inativos"], 1)
        bad = deepcopy(base)
        bad["eixos"][0]["documentos_ids"] = ["unknown"]
        before = self.snapshot()
        with self.assertRaises(ValueError):
            prepare_base_import(encoded(bad))
        self.assertEqual(before, self.snapshot())
        imported_again = import_prepared_base(self.project, prepare_base_import(encoded(fixture())))
        self.assertFalse(imported_again["root_metadata_preserved"])
        self.assertEqual(self.project.base()["eixos"], base["eixos"])

    def test_legacy_vectors_separate_profile_preserves_destination_active_profile(self):
        cloud_profile = {"perfil_id": "pf_cloud_gemma", "provedor": "ollama",
                         "modelo": "embeddinggemma", "dimensoes": 768}
        self.project.save_config({"perfis": {"pf_cloud_gemma": cloud_profile}, "perfil_ativo_id": "pf_cloud_gemma"})
        prepared = prepare_base_import(encoded(fixture()), legacy_npz(fixture()))
        result = import_prepared_base(self.project, prepared)
        config = self.project.config()
        self.assertEqual(config["perfil_ativo_id"], "pf_cloud_gemma")
        self.assertEqual(config["perfis"]["pf_cloud_gemma"], cloud_profile)
        legacy = config["perfis"][result["legacy_profile_id"]]
        self.assertEqual(legacy["provedor"], "openai")
        self.assertTrue(legacy["legado"])
        self.assertEqual(result["legacy_embeddings"], 2)
        manifest, matrix, _ = self.project.load_index(legacy["perfil_id"], CATEGORIES[0])
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual({i["topico_id"] for i in manifest["itens"]}, {"topico_1", "topico_2"})
        self.assertTrue(all(u["status"] == "atualizado" for t in self.project.base()["documentos"][0]["estrutura"][:2]
                            for c in t["embeddings"]["perfis"][legacy["perfil_id"]]["categorias"].values()
                            for u in c["unidades"]))

    def test_upsert_preserves_unrelated_destination_document(self):
        old = fixture()
        old["documentos"][0]["documento_id"] = "cloud_only"
        for t in old["documentos"][0]["estrutura"]:
            t["topico_id"] = "cloud_"+t["topico_id"]
        import_prepared_base(self.project, prepare_base_import(encoded(old)))
        result = import_prepared_base(self.project, prepare_base_import(encoded(fixture())))
        self.assertEqual(result["total_documents"], 2)
        self.assertEqual(result["total_topics"], 6)
        self.assertEqual(len(list((self.project.root/"historico"/"bases").glob("*.json"))), 1)

    def test_duplicate_ids_invalid_shapes_credentials_and_nonfinite_reject_before_write(self):
        bad_bases = []
        b = fixture(); b["documentos"].append(deepcopy(b["documentos"][0])); bad_bases.append(b)
        b = fixture(); b["documentos"][0]["estrutura"][1]["topico_id"] = "topico_1"; bad_bases.append(b)
        b = fixture(); b["documentos"][0]["estrutura"] = {}; bad_bases.append(b)
        b = fixture(); b["documentos"][0]["metadados"]["eixos"] = [5]; bad_bases.append(b)
        b = fixture(); b["documentos"][0]["estrutura"][0]["ativo"] = "false"; bad_bases.append(b)
        b = fixture(); b["documentos"][0]["estrutura"][0]["perguntas_sugeridas"] = "texto"; bad_bases.append(b)
        b = fixture(); b["documentos"][0]["metadados"]["api_key"] = "must-not-store"; bad_bases.append(b)
        b = fixture(); b["documentos"][0]["metadados"]["value"] = float("nan"); bad_bases.append(b)
        b = fixture(); b["documentos"][0]["documento_id"] = "../server"; bad_bases.append(b)
        before = self.snapshot()
        for value in bad_bases:
            with self.subTest(value=value["documentos"][0].get("documento_id")):
                with self.assertRaises(ValueError):
                    prepare_base_import(encoded(value))
                self.assertEqual(before, self.snapshot())
        for raw in [b'{"documentos":[],"documentos":[]}', b'[]', b'not json']:
            with self.assertRaises(ValueError):
                prepare_base_import(raw)

    def test_destination_topic_collision_is_rejected_without_changing_any_file(self):
        old = fixture()
        import_prepared_base(self.project, prepare_base_import(encoded(old)))
        different_doc = fixture()
        different_doc["documentos"][0]["documento_id"] = "other_document"
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "topico_id duplicado"):
            import_prepared_base(self.project, prepare_base_import(encoded(different_doc)))
        self.assertEqual(before, self.snapshot())

    def test_invalid_npz_ids_dimensions_hashes_and_values_reject_before_write(self):
        cases = [
            {"vetores": np.asarray([[float("nan"), 0, 0], [0, 1, 0]])},
            {"vetores": np.zeros((2, 3))},
            {"vetores": np.asarray([[1e308, 0, 0], [0, 1, 0]])},
            {"dimensoes": np.asarray([4])},
            {"topico_ids": np.asarray(["topico_1", "topico_1"])},
            {"topico_ids": np.asarray(["topico_1", "unknown"])},
            {"hashes_texto": np.asarray(["f"*64, "f"*64])},
            {"modelo": np.asarray(["not-an-openai-model"])},
            {"modelo": np.asarray(["text-embedding-3-small"], dtype=object)},
        ]
        before = self.snapshot()
        for changes in cases:
            with self.subTest(fields=list(changes)):
                with self.assertRaises(ValueError):
                    prepare_base_import(encoded(fixture()), legacy_npz(fixture(), **changes))
                self.assertEqual(before, self.snapshot())

    def test_npz_extra_paths_expansion_limit_and_forged_shape_are_rejected(self):
        valid = legacy_npz(fixture())
        stream = io.BytesIO(valid)
        with zipfile.ZipFile(stream, "a") as package:
            package.writestr("../server.npy", b"bad")
        with self.assertRaises(ValueError):
            prepare_base_import(encoded(fixture()), stream.getvalue())
        with patch("autorag.base_import.MAX_EXPANDED_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "64 MiB"):
                prepare_base_import(encoded(fixture()), valid)
        rewritten = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(valid)) as source, zipfile.ZipFile(rewritten, "w") as target:
            for name in source.namelist():
                if name == "vetores.npy":
                    header = "{'descr': '<f8', 'fortran_order': False, 'shape': (999999999, 3), }"
                    header += " " * ((64 - ((len(header)+11) % 64)) % 64) + "\n"
                    target.writestr(name, np.lib.format.magic(1, 0)+len(header).to_bytes(2, "little")+header.encode())
                else:
                    target.writestr(name, source.read(name))
        with self.assertRaisesRegex(ValueError, "Forma NPY"):
            prepare_base_import(encoded(fixture()), rewritten.getvalue())

    def test_publish_failure_rolls_back_active_base_config_and_pointer(self):
        import_prepared_base(self.project, prepare_base_import(encoded(fixture())))
        before_base = self.project.base_path.read_bytes()
        before_config = self.project.config_path.read_bytes() if self.project.config_path.exists() else None
        prepared = prepare_base_import(encoded(fixture()), legacy_npz(fixture()))
        with patch.object(self.project, "publish", side_effect=OSError("disco indisponível")):
            with self.assertRaises(OSError):
                import_prepared_base(self.project, prepared)
        self.assertEqual(self.project.base_path.read_bytes(), before_base)
        self.assertEqual(self.project.config_path.read_bytes() if self.project.config_path.exists() else None, before_config)


class BaseTransferHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Project(Path(self.temp.name)/"project")
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.patches = [
            patch.object(api, "project", self.project), patch.object(api, "executor", self.worker),
            patch.object(api, "jobs", {}), patch.dict(os.environ, {"AUTORAG_API_TOKEN": "transfer-test-token", "AUTORAG_ENV": "development"}),
        ]
        for p in self.patches: p.start()
        self.context = TestClient(api.app, headers={"Authorization": "Bearer transfer-test-token"})
        self.client = self.context.__enter__()

    def tearDown(self):
        self.worker.shutdown(wait=True)
        self.context.__exit__(None, None, None)
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def wait(self, response):
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["job_id"]
        end = time.monotonic()+8
        while time.monotonic() < end:
            job = self.client.get("/api/jobs/"+job_id).json()
            if job["status"] in {"failed", "succeeded"}:
                return job
            time.sleep(.01)
        self.fail("Job de transferência não terminou.")

    def upload(self, base=None, npz=None):
        files = {"base_file": ("../../ignored.json", encoded(base or fixture()), "application/json")}
        if npz is not None: files["legacy_npz"] = ("ignored.npz", npz, "application/octet-stream")
        return self.client.post("/api/base/import", files=files)

    def test_import_export_auth_and_explicit_token_required_even_on_loopback(self):
        for route, method in [("/api/base/export", "get"), ("/api/base/import", "post"), ("/api/indices/import", "post")]:
            self.assertEqual(getattr(self.client, method)(route, headers={"Authorization": ""}).status_code, 401)
            with patch.dict(os.environ, {"AUTORAG_API_TOKEN": ""}):
                self.assertEqual(getattr(self.client, method)(route).status_code, 403)
        self.assertEqual(self.client.get("/api/base/export").status_code, 404)

    def test_structured_import_and_export_preserve_complete_document_content(self):
        job = self.wait(self.upload(npz=legacy_npz(fixture())))
        self.assertEqual(job["status"], "succeeded", job)
        result = job["result"]
        response = self.client.get("/api/base/export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(canonical_digest(response.json()), canonical_digest(fixture()))
        self.assertEqual(response.headers["X-AutoRAG-Canonical-Digest"], result["canonical_digest"])
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(result["legacy_embeddings"], 2)

    def test_rejects_invalid_payload_before_enqueuing_or_changing_base(self):
        before = list(self.project.root.rglob("*"))
        b = fixture(); b["documentos"][0]["estrutura"][1]["topico_id"] = "topico_1"
        self.assertEqual(self.upload(b).status_code, 422)
        self.assertEqual(api.jobs, {})
        self.assertEqual(list(self.project.root.rglob("*")), before)
        self.assertEqual(self.client.post("/api/base/import").status_code, 422)

    def test_cloud_adapter_checkpoints_before_job_succeeds(self):
        from cloudflare.adapter import install
        events = []
        class Store:
            def save_job(self, record): events.append(("job", record["status"]))
            def checkpoint(self, project): events.append(("checkpoint", None))
        with patch.object(api, "submit", api.submit):
            install(api, Store())
            job = self.wait(self.upload())
        self.assertEqual(job["status"], "succeeded", job)
        self.assertLess(events.index(("checkpoint", None)), events.index(("job", "succeeded")))

    def test_native_index_route_validates_before_enqueue_and_uses_existing_queue(self):
        with patch("autorag.index_transfer.validate_index_bundle", return_value={"vectors": 2}) as validate, \
             patch("autorag.index_transfer.import_index_bundle", return_value={"vectors": 2, "committed": True}) as publish:
            job = self.wait(self.client.post("/api/indices/import", files={"index_file": ("bundle.zip", b"zip", "application/zip")}))
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(validate.call_args.args, (self.project, b"zip"))
        self.assertEqual(publish.call_args.args, (self.project, b"zip"))
        with patch("autorag.index_transfer.validate_index_bundle", side_effect=ValueError("bundle inválido")), \
             patch("autorag.index_transfer.import_index_bundle") as publish:
            response = self.client.post("/api/indices/import", files={"index_file": ("bundle.zip", b"bad", "application/zip")})
        self.assertEqual(response.status_code, 422)
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
