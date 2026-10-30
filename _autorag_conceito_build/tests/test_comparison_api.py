"""HTTP tests for saved experiments, comparison downloads and optional evaluation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
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

from autorag.core import Project, write_json
from autorag.service import generate, import_base
from test_autorag import FakeProvider, fixture

WEB_AVAILABLE = bool(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"))
if WEB_AVAILABLE:
    from fastapi.testclient import TestClient
    import api


@unittest.skipUnless(WEB_AVAILABLE, "Instale requirements-web.txt.")
class ComparisonAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = Project(self.root/"project")
        source = self.root/"base.json"
        write_json(source, fixture())
        import_base(self.project, source)
        self.profile = {"perfil_id": "pf_test", "provedor": "teste", "modelo": "offline",
                        "dimensoes": 3, "revisao_modelo": "fixed", "limite_tokens": 512,
                        "normalizacao": "l2", "versao_preparacao": "1"}
        self.project.save_config({"perfis": {"pf_test": self.profile}, "perfil_ativo_id": "pf_test"})
        self.provider = FakeProvider(self.profile)
        generate(self.project, "pf_test", ["conteudo_fonte", "resumo"], provider=self.provider)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.patches = [
            patch.object(api, "project", self.project), patch.object(api, "executor", self.worker),
            patch.object(api, "jobs", {}), patch("autorag.experiments.get_provider", return_value=self.provider),
            patch("autorag.assistance.make_structured_generator", side_effect=api.LLMError("Apoio simulado indisponível neste teste.")),
            patch.dict(os.environ, {"AUTORAG_API_TOKEN": "test-token-for-comparison-http-with-32-characters",
                                    "AUTORAG_ENV": "development"})]
        for item in self.patches:
            item.start()
        self.context = TestClient(api.app, headers={"Authorization": "Bearer test-token-for-comparison-http-with-32-characters"})
        self.client = self.context.__enter__()

    def tearDown(self):
        self.worker.shutdown(wait=True)
        self.context.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def config(self, name="Experimento", **extra):
        return {"name": name, "profile_ids": ["pf_test"], "mode": "vector",
                "categories": ["conteudo_fonte"], "top_k": 2, **extra}

    def wait_job(self, response, expected="succeeded"):
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.client.get("/api/jobs/" + job_id).json()
            if job["status"] in {"succeeded", "failed"}:
                self.assertEqual(job["status"], expected, job)
                return job["result"] if expected == "succeeded" else job
            time.sleep(0.01)
        self.fail("O job não concluiu.")

    def test_configuration_save_reload_and_use_as_consultation_parameters(self):
        response = self.client.post("/api/experiment-configs", json={
            "name": "Resumo e original", "experiments": [self.config(categories=["resumo"])], "active": True})
        self.assertEqual(response.status_code, 200, response.text)
        preset = response.json()
        self.assertEqual(self.client.get("/api/experiment-configs").json()["configs"], [preset])
        current = self.client.get("/api/consultation-config").json()
        self.assertEqual(current["experiments"], preset["experiments"])
        result = self.wait_job(self.client.post("/api/search", json={"query": "Consulta normativa"}))
        self.assertEqual(result["experiments"][0]["config"]["categories"], ["resumo"])
        self.assertEqual(result["analysis"]["status"], "disabled")
        self.assertEqual(result["analysis_config"]["model"], "gpt-6-luna")

    def test_independent_modes_save_run_export_and_reuse_through_http(self):
        from test_independent_retrieval import projection
        from test_semantic import fake_model
        configs = [self.config("Somente UMAP", mode="umap", metric="euclidean", n_components=2),
                   self.config("Somente HDBSCAN", mode="hdbscan", min_cluster_size=2, dendrogram=True)]
        with patch("autorag.semantic._fit_umap", side_effect=projection), \
                patch("autorag.semantic._fit_hdbscan", side_effect=lambda matrix, params: fake_model([0]*len(matrix))):
            saved = self.client.post("/api/experiment-configs", json={"name": "Etapas independentes", "experiments": configs})
            self.assertEqual(saved.status_code, 200, saved.text)
            result = self.wait_job(self.client.post("/api/comparisons", json={"query": "Consulta normativa", "preset_id": saved.json()["id"]}))
            self.assertEqual([e["config"]["mode"] for e in result["experiments"]], ["umap", "hdbscan"])
            self.assertTrue(all(e["status"] == "ok" for e in result["experiments"]))
            reproduced = self.wait_job(self.client.post("/api/comparisons/"+result["comparison_id"]+"/reproduce", json={}))
            self.assertEqual(result["experiments"], reproduced["experiments"])
            for config in configs:
                legacy = self.wait_job(self.client.post("/api/search", json={
                    key: value for key, value in {**config, "profile_id": "pf_test", "query": "Consulta normativa"}.items()
                    if key not in {"name", "profile_ids"}}))
                self.assertEqual(legacy["status"], "ok")
                self.assertEqual(legacy["mode"], config["mode"])
        filtered = self.wait_job(self.client.post("/api/search", json={
            "query": "Consulta normativa", "profile_id": "pf_test", "mode": "vector", "minimum": 1.0}))
        self.assertEqual(filtered["semantic_results"], [])
        response = self.client.get("/api/comparisons/"+result["comparison_id"]+"/export?format=vectors")
        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        manifest = json.loads(archive.read("manifesto.json"))
        self.assertEqual(len(manifest["projections"]), 1)
        self.assertTrue(any(row["distance_space"] == "umap_projection" for row in manifest["associations"]))

    def test_no_config_uses_pure_original_content_vectors(self):
        result = self.wait_job(self.client.post("/api/search", json={"query": "Consulta normativa"}))
        self.assertEqual(result["experiments"][0]["config"]["mode"], "vector")
        self.assertEqual(result["experiments"][0]["config"]["categories"], ["conteudo_fonte"])
        self.assertEqual(self.client.get("/health").json()["version"], "0.3.0")

    def test_three_experiments_persist_and_download_all_formats(self):
        result = self.wait_job(self.client.post("/api/comparisons", json={
            "query": "Consulta normativa", "experiments": [
                self.config("Cosseno"), self.config("Euclidiana", metric="euclidean"),
                self.config("Resumo", categories=["resumo"])]}))
        self.assertEqual(len(result["experiments"]), 3)
        comparison_id = result["comparison_id"]
        listed = self.client.get("/api/comparisons").json()["comparisons"]
        self.assertEqual(listed[0]["comparison_id"], comparison_id)
        self.assertEqual(self.client.get("/api/comparisons/" + comparison_id).json(), result)
        for format in ("json", "jsonl", "csv", "config"):
            response = self.client.get("/api/comparisons/" + comparison_id + "/export", params={"format": format})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("attachment", response.headers["content-disposition"])
            self.assertTrue(response.content)
        config = self.client.get("/api/comparisons/" + comparison_id + "/export?format=config").json()
        self.assertEqual(len(config["experiments"]), 3)
        self.assertFalse(config["analysis"]["enabled"])

    def test_successful_analysis_is_saved_in_jsonl_csv_and_each_original_result_row(self):
        def answer(url, payload, key=""):
            inputs = json.loads(payload["input"])
            return {"status": "completed", "output": [{"content": [{"type": "output_text", "text": json.dumps({
                "assessments": [{"item_id": item["item_id"], "classification": "pertinente", "relevant": True,
                                 "comment": "O conteúdo é útil para a consulta.", "utility_rank": number}
                                for number, item in enumerate(inputs["items"], 1)]})}]}]}
        with patch("autorag.llm.obter_chave", return_value="private-key"), \
                patch("autorag.llm._request", side_effect=answer):
            result = self.wait_job(self.client.post("/api/comparisons", json={
                "query": "Consulta normativa", "experiments": [self.config()],
                "analysis": {"enabled": True, "provider": "openai", "model": "gpt-6-luna"}}))
        self.assertEqual(result["analysis"]["status"], "succeeded")
        comparison_id = result["comparison_id"]
        stored = self.client.get("/api/comparisons/" + comparison_id).json()
        self.assertEqual(stored, result)
        lines = self.client.get("/api/comparisons/" + comparison_id + "/export?format=jsonl")
        first = json.loads(lines.text.splitlines()[0])
        self.assertEqual(first["analysis"], result["experiments"][0]["results"][0]["analysis"])
        self.assertTrue(first["analysis"]["relevant"])
        csv = self.client.get("/api/comparisons/" + comparison_id + "/export?format=csv")
        self.assertIn("O conteúdo é útil", csv.text)
        self.assertNotIn("private-key", lines.text)
        self.assertEqual(stored["analysis_config"]["model"], "gpt-6-luna")
        vectors = self.client.get("/api/comparisons/" + comparison_id + "/export?format=vectors")
        self.assertEqual(vectors.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(vectors.content)) as archive:
            manifest = json.loads(archive.read("manifesto.json"))
            assessed = [item["analysis"] for item in manifest["associations"]
                        if isinstance(item.get("analysis"), dict) and item["analysis"].get("status") == "succeeded"]
            self.assertTrue(assessed, "O ZIP deve incluir a análise final, não apenas a recuperação inicial.")
            self.assertEqual(assessed[0]["comment"], "O conteúdo é útil para a consulta.")

    def test_failed_analysis_still_finishes_retrieval_and_exports_unclassified_rows(self):
        with patch("autorag.llm.obter_chave", return_value=""):
            result = self.wait_job(self.client.post("/api/comparisons", json={
                "query": "Consulta normativa", "experiments": [self.config()],
                "analysis": {"enabled": True, "provider": "openai", "model": "gpt-6-luna"}}))
        self.assertEqual(result["analysis"]["status"], "failed")
        self.assertTrue(result["experiments"][0]["results"])
        self.assertNotIn("classification", result["experiments"][0]["results"][0]["analysis"])
        response = self.client.get("/api/comparisons/" + result["comparison_id"] + "/export?format=jsonl")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.text.splitlines()[0])["analysis"]["status"], "failed")

    def test_preset_can_reproduce_named_experiments_and_analysis_selection(self):
        saved = self.client.post("/api/experiment-configs", json={
            "name": "Reproduzível", "experiments": [self.config("Nome persistido")],
            "analysis": {"enabled": False, "provider": "ollama", "model": "llama:test"}}).json()
        result = self.wait_job(self.client.post("/api/comparisons", json={
            "query": "Consulta normativa", "preset_id": saved["id"]}))
        self.assertEqual(result["experiments"][0]["config"]["name"], "Nome persistido")
        self.assertEqual(result["analysis_config"]["model"], "llama:test")

    def test_clear_consultation_config_restores_fallback_and_reject_unknown_settings(self):
        response = self.client.put("/api/consultation-config", json={"experiments": [],
            "analysis": {"enabled": False}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["experiments"], [])
        for request in (
                {"query": "Consulta", "experiments": [self.config()] * 4},
                {"query": "Consulta", "experiments": [self.config()], "api_key": "secret"},
                {"query": "Consulta", "experiments": [{**self.config(), "endpoint": "https://bad.test"}]},
                {"query": "Consulta", "experiments": [self.config()], "analysis": {"enabled": True, "api_key": "secret"}},
                {"query": " "},
                {"query": "Consulta", "experiments": [self.config(), self.config()]}):
            with self.subTest(request=request):
                response = self.client.post("/api/comparisons", json=request)
                self.assertIn(response.status_code, {400, 422}, response.text)
        self.assertEqual(self.client.post("/api/search", json={"query": "Consulta", "top_k": 3}).status_code, 422)

    def test_download_ids_and_authentication_cannot_bypass_limits(self):
        for path in ("/api/comparisons/comparison_missing", "/api/comparisons/comparison_missing/export",
                     "/api/comparisons/not-valid", "/api/comparisons/not-valid/export"):
            self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.get("/api/comparisons").status_code, 200)
        self.assertEqual(self.client.get("/api/comparisons", headers={"Authorization": ""}).status_code, 401)
        self.assertEqual(self.client.get("/api/experiment-configs", headers={"Authorization": ""}).status_code, 401)
        result = self.wait_job(self.client.post("/api/comparisons", json={"query": "Consulta"}))
        self.assertEqual(self.client.get("/api/comparisons/" + result["comparison_id"] + "/export?format=zip").status_code, 422)

    def test_strict_reproduction_reuses_saved_vectors_and_never_calls_embedding_or_llm(self):
        original = self.wait_job(self.client.post("/api/comparisons", json={
            "query": "Consulta normativa", "experiments": [self.config()]}))
        with patch("autorag.experiments.get_provider", side_effect=AssertionError("Inferência não autorizada")) as embedding, \
                patch("autorag.llm._request", side_effect=AssertionError("LLM não autorizado")) as llm:
            reproduced = self.wait_job(self.client.post(
                "/api/comparisons/" + original["comparison_id"] + "/reproduce"))
        self.assertNotEqual(reproduced["comparison_id"], original["comparison_id"])
        self.assertEqual(reproduced["reproduced_from"], original["comparison_id"])
        self.assertEqual(reproduced["experiments"][0]["results"], original["experiments"][0]["results"])
        self.assertEqual(reproduced["base_checksum"], original["base_checksum"])
        self.assertEqual(reproduced["analysis"]["status"], "disabled")
        self.assertFalse(reproduced["analysis_config"]["enabled"])
        embedding.assert_not_called()
        llm.assert_not_called()
        self.assertEqual(self.client.get("/api/comparisons/" + reproduced["comparison_id"]).status_code, 200)
        self.assertEqual(self.client.post("/api/comparisons/" + original["comparison_id"] + "/reproduce",
                         headers={"Authorization": ""}).status_code, 401)
        self.assertEqual(self.client.post("/api/comparisons/comparison_missing/reproduce").status_code, 404)

    def test_reproduction_rejects_modified_or_missing_artifact_without_new_inference(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                original = self.wait_job(self.client.post("/api/comparisons", json={
                    "query": "Consulta " + str(missing), "experiments": [self.config()]}))
                reference = original["experiments"][0]["spaces"][0]["input_reference"]["arquivo_npz"]
                saved = self.project.path(reference)
                if missing:
                    saved.unlink()
                else:
                    saved.write_bytes(saved.read_bytes() + b"changed")
                with patch("autorag.experiments.get_provider") as embedding, patch("autorag.llm._request") as llm:
                    failed = self.wait_job(self.client.post(
                        "/api/comparisons/" + original["comparison_id"] + "/reproduce"), expected="failed")
                self.assertTrue(failed["error"])
                self.assertNotIn("result", failed)
                self.assertNotIn(str(self.project.root), failed["error"])
                embedding.assert_not_called()
                llm.assert_not_called()

    def test_vector_download_contains_saved_query_matches_and_metrics(self):
        result = self.wait_job(self.client.post("/api/comparisons", json={
            "query": "Consulta normativa", "experiments": [self.config()]}))
        comparison_id = result["comparison_id"]
        path = self.project.root/"estudos"/"comparacoes"/comparison_id/"vetores.zip"
        self.assertTrue(path.is_file(), "O ZIP deve ser salvo antes de concluir a tarefa.")
        saved = path.read_bytes()
        response = self.client.get("/api/comparisons/" + comparison_id + "/export?format=vectors")
        self.assertEqual(response.status_code, 200, response.text if response.status_code != 200 else "")
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertIn(comparison_id + "-vetores.zip", response.headers["content-disposition"])
        self.assertTrue(response.content.startswith(b"PK"))
        self.assertEqual(response.content, saved)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            matrices = [name for name in names if name.endswith(".npz")]
            self.assertGreaterEqual(len(matrices), 2)
            self.assertTrue(any(name.endswith(".csv") for name in names))
            self.assertTrue(any(name.endswith(".json") for name in names))
            for name in matrices:
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as package:
                    matrix = package["vetores"]
                    self.assertEqual(matrix.shape[1], self.profile["dimensoes"])
                    self.assertTrue(np.isfinite(matrix).all())
            csv_text = "\n".join(archive.read(name).decode("utf-8-sig") for name in names if name.endswith(".csv"))
            self.assertIn("topic_id", csv_text)
            self.assertIn("distance", csv_text)
            self.assertIn(result["experiments"][0]["results"][0]["topic_id"], csv_text)

    def test_vector_download_requires_auth_and_rejects_changed_query(self):
        result = self.wait_job(self.client.post("/api/comparisons", json={"query": "Consulta vetorial"}))
        comparison_id = result["comparison_id"]
        with patch("autorag.comparisons.export_vector_bundle") as bundle:
            response = self.client.get("/api/comparisons/" + comparison_id + "/export?format=vectors",
                                       headers={"Authorization": ""})
        self.assertEqual(response.status_code, 401)
        bundle.assert_not_called()
        reference = result["experiments"][0]["spaces"][0]["input_reference"]["arquivo_npz"]
        path = self.project.path(reference)
        path.write_bytes(path.read_bytes() + b"changed")
        response = self.client.get("/api/comparisons/" + comparison_id + "/export?format=vectors")
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("alterado", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
