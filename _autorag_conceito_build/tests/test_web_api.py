"""HTTP integration tests with isolated projects and no model/network calls."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from autorag import core as core_module
from autorag.core import CATEGORIES, Project, read_json, write_json
from autorag.web_service import DEFAULT_CATEGORIES, present_results

WEB_AVAILABLE = bool(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx")
                     and importlib.util.find_spec("multipart"))
if WEB_AVAILABLE:
    import httpx
    from fastapi.testclient import TestClient
    import api

TOKEN = "test-only-token-with-over-32-characters"


@unittest.skipUnless(WEB_AVAILABLE, "Instale requirements-web.txt para os testes HTTP.")
class WebAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Project(Path(self.temp.name)/"project")
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.worker_errors = []
        real_submit = self.worker.submit
        def tracked_submit(*args, **kwargs):
            future = real_submit(*args, **kwargs)
            def record_failure(done):
                if done.exception() is not None:
                    self.worker_errors.append(repr(done.exception()))
            future.add_done_callback(record_failure)
            return future
        self.worker.submit = tracked_submit
        self.patches = [
            patch.object(api, "project", self.project),
            patch.object(api, "executor", self.worker),
            patch.object(api, "jobs", {}),
            patch.dict(os.environ, {"AUTORAG_API_TOKEN": TOKEN, "AUTORAG_ENV": "development",
                                    "AUTORAG_ALLOWED_SOURCE_ROOTS": ""}),
        ]
        for p in self.patches:
            p.start()
        self.context = TestClient(api.app, headers={"Authorization": "Bearer "+TOKEN})
        self.client = self.context.__enter__()

    def tearDown(self):
        # No test may leave a job using another test's global project.
        self.worker.shutdown(wait=True)
        self.context.__exit__(None, None, None)
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def wait_for_job(self, response):
        self.assertEqual(response.status_code, 202, response.text)
        receipt = response.json()
        self.assertRegex(receipt["job_id"], r"^job_[A-Za-z0-9_]+$")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = self.client.get("/api/jobs/"+receipt["job_id"])
            self.assertEqual(status.status_code, 200, status.text)
            job = status.json()
            if job["status"] in {"succeeded", "failed"}:
                self.assertIn("finished_at", job)
                return job
            self.assertIn(job["status"], {"queued", "running"})
            time.sleep(0.01)
        self.fail(f"A tarefa local não terminou: {job}; worker errors={self.worker_errors}")

    def upload(self, *, commit=False, metadata=None, filename="norma.md", content=None):
        return self.client.post("/api/ingest", data={
            "sources": "[]", "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            "commit": str(commit).lower(),
        }, files={"files": (filename, content or "# Compras\n\nArt. 1º Mantenha os registros públicos.".encode(), "text/markdown")})

    def test_authentication_and_health(self):
        self.assertEqual(self.client.get("/health", headers={"Authorization": ""}).status_code, 200)
        for authorization in ["", "Bearer wrong", TOKEN]:
            response = self.client.get("/api/status", headers={"Authorization": authorization})
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.get("/api/status").status_code, 200)

    def test_remote_client_requires_configured_token(self):
        async def request():
            transport = httpx.ASGITransport(app=api.app, client=("203.0.113.15", 1234))
            async with httpx.AsyncClient(transport=transport, base_url="http://server") as client:
                return await client.get("/api/status")
        with patch.dict(os.environ, {"AUTORAG_API_TOKEN": ""}):
            self.assertEqual(self.client.get("/api/status").status_code, 200)
            self.assertEqual(asyncio.run(request()).status_code, 403)

    def test_production_rejects_short_token_at_startup(self):
        with patch.dict(os.environ, {"AUTORAG_ENV": "production", "AUTORAG_API_TOKEN": "short"}):
            with self.assertRaisesRegex(RuntimeError, "32"):
                with TestClient(api.app):
                    pass

    def test_status_is_generic_and_exposes_defaults_and_categories(self):
        response = self.client.get("/api/status")
        data = response.json()
        self.assertEqual(data["documents"], 0)
        self.assertEqual(data["topics"], 0)
        self.assertEqual(data["default_top_k"], 6)
        self.assertEqual(data["default_categories"], list(DEFAULT_CATEGORIES))
        self.assertEqual({r["id"] for r in data["categories"]}, set(CATEGORIES))
        self.assertFalse(data["local_paths_enabled"])
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_request_size_is_bounded_for_declared_and_chunked_bodies(self):
        with patch.object(api, "MAX_REQUEST_BYTES", 32):
            for body in [b"x"*40, iter([b"x"*20, b"x"*20])]:
                response = self.client.post("/api/search", content=body,
                                            headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 413, response.text)

    def test_upload_preview_is_independent_of_metadata_files(self):
        job = self.wait_for_job(self.upload(metadata={
            "title": "Código de compras", "orgao": "Tribunal de Contas",
            "eixo": "Contratações", "idioma": "português",
        }))
        self.assertEqual(job["status"], "succeeded", job)
        result = job["result"]
        self.assertFalse(result["committed"])
        self.assertEqual(result["documents"], 1)
        self.assertGreater(result["topics"], 0)
        self.assertFalse(self.project.base_path.exists())
        preview = result["preview"][0]
        self.assertEqual(preview["title"], "Código de compras")
        self.assertEqual(preview["metadata"]["metadados"]["orgao"], "Tribunal de Contas")
        self.assertEqual(preview["metadata"]["metadados"]["idioma"], "português")
        self.assertTrue(preview["topics"][0]["processamento"]["resumo_extrativo"])
        self.assertTrue(result["warnings"])
        self.assertRegex(result["preview_id"], r"^preview_")

    def test_preview_commit_uses_reviewed_snapshot_and_preserves_metadata(self):
        preview = self.wait_for_job(self.upload(metadata={"title": "Regras", "eixo": "Gestão"}))["result"]
        with patch("autorag.ingestion.process_upload", side_effect=AssertionError("Não reprocessar prévia.")):
            job = self.wait_for_job(self.client.post("/api/ingest", data={
                "commit": "true", "preview_id": preview["preview_id"],
            }))
        self.assertEqual(job["status"], "succeeded", job)
        self.assertTrue(job["result"]["committed"])
        base = self.project.base()
        self.assertEqual(len(base["documentos"]), 1)
        self.assertEqual(base["documentos"][0]["nome_documento"], "Regras")
        self.assertEqual(base["eixos"][0]["nome"], "Gestão")
        self.assertEqual(self.client.get("/api/status").json()["documents"], 1)

    def test_commit_upserts_same_source_and_keeps_backup(self):
        first = self.wait_for_job(self.upload(commit=True, metadata={"orgao": "A"}))
        self.assertEqual(first["status"], "succeeded", first)
        before = self.project.base()["documentos"][0]
        second = self.wait_for_job(self.upload(commit=True, metadata={"orgao": "B"}))
        self.assertEqual(second["status"], "succeeded", second)
        base = self.project.base()
        self.assertEqual(len(base["documentos"]), 1)
        self.assertEqual(base["documentos"][0]["documento_id"], before["documento_id"])
        self.assertEqual(base["documentos"][0]["estrutura"][0]["topico_id"], before["estrutura"][0]["topico_id"])
        self.assertEqual(base["documentos"][0]["metadados"]["orgao"], "B")
        self.assertTrue(list((self.project.root/"historico"/"bases").glob("*.json")))

    def test_duplicate_uploaded_names_fail_without_mutating_base(self):
        response = self.client.post("/api/ingest", data={"commit": "true"}, files=[
            ("files", ("same.txt", b"Regra um.", "text/plain")),
            ("files", ("same.txt", b"Regra dois.", "text/plain")),
        ])
        job = self.wait_for_job(response)
        self.assertEqual(job["status"], "failed")
        self.assertIn("repetidas", job["error"])
        self.assertFalse(self.project.base_path.exists())

    def test_duplicate_local_sources_fail_and_local_paths_require_roots(self):
        source = Path(self.temp.name)/"source.txt"
        source.write_text("Norma administrativa.", encoding="utf-8")
        failed = self.wait_for_job(self.client.post("/api/ingest", data={"sources": json.dumps([str(source)])}))
        self.assertEqual(failed["status"], "failed")
        self.assertIn("desativados", failed["error"])
        with patch.dict(os.environ, {"AUTORAG_ALLOWED_SOURCE_ROOTS": self.temp.name}):
            self.assertTrue(self.client.get("/api/status").json()["local_paths_enabled"])
            job = self.wait_for_job(self.client.post("/api/ingest", data={
                "sources": json.dumps([str(source), str(source)]),
            }))
            self.assertEqual(job["status"], "failed")
            self.assertIn("duplicado", job["error"])

    def test_ingestion_validates_form_and_preview_ids(self):
        cases = [
            {}, {"sources": "not-json"}, {"metadata": "[]"},
            {"sources": json.dumps([123])},
            {"sources": json.dumps(["a"]*31)},
            {"preview_id": "preview_../../outside", "commit": "true"},
            {"preview_id": "preview_missing", "commit": "true"},
            {"preview_id": "preview_missing", "commit": "false"},
        ]
        for data in cases:
            with self.subTest(data=data):
                self.assertEqual(self.client.post("/api/ingest", data=data).status_code, 422)

    def test_search_validates_categories_and_parameter_limits(self):
        valid = {"query": "Consulta", "profile_id": "pf_test"}
        for update in [
            {"categories": ["desconhecida"]}, {"categories": []}, {"mode": "unknown"},
            {"top_k": 0}, {"top_k": 101}, {"metric": "unknown"},
            {"n_neighbors": 1}, {"min_dist": -1}, {"dendrogram": "not-bool"},
            {"query": ""}, {"query": "x"*50001}, {"unknown_option": 1},
        ]:
            with self.subTest(update=update):
                self.assertEqual(self.client.post("/api/search", json={**valid, **update}).status_code, 422)

    def test_search_job_receives_defaults_without_paid_provider(self):
        output = {"status": "query_noise", "semantic_results": [], "baseline_results": []}
        with patch.object(api, "run_search", return_value=output) as search:
            job = self.wait_for_job(self.client.post("/api/search", json={
                "query": "Que registros devem ser mantidos?", "profile_id": "pf_test",
            }))
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["result"], output)
        request = search.call_args.args[1]
        self.assertIs(search.call_args.args[0], self.project)
        self.assertEqual(request["categories"], list(DEFAULT_CATEGORIES))
        self.assertEqual(request["mode"], "semantic")
        self.assertEqual(request["top_k"], 6)
        self.assertFalse(request["dendrogram"])

    def test_laboratory_controls_are_forwarded_independently(self):
        request = {"query": "Consulta de licitação", "profile_id": "pf_test", "mode": "laboratory",
                   "categories": ["palavras_chave", "conteudo_fonte"], "metric": "euclidean",
                   "top_k": 4, "view": "vetores", "dendrogram": True,
                   "minimum": 0.1, "maximum": 0.9, "partial": True,
                   "individual_limits": {"conteudo_fonte": 0.2}, "individual_rule": "qualquer"}
        with patch.object(api, "run_search", return_value={"status": "ok"}) as search:
            job = self.wait_for_job(self.client.post("/api/search", json=request))
        self.assertEqual(job["status"], "succeeded", job)
        forwarded = search.call_args.args[1]
        for key, value in request.items():
            self.assertEqual(forwarded[key], value)

    def test_job_failure_remains_available_after_worker_finishes(self):
        with patch.object(api, "run_search", side_effect=ValueError("Categorias sem embeddings atuais.")):
            job = self.wait_for_job(self.client.post("/api/search", json={
                "query": "Consulta", "profile_id": "pf_test",
            }))
        # A terminal API snapshot can be visible before its final disk write.
        self.worker.shutdown(wait=True)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "Categorias sem embeddings atuais.")
        self.assertEqual(read_json(self.project.root/"jobs"/(job["job_id"]+".json"))["status"], "failed")
        self.assertEqual(self.client.get("/api/jobs/"+job["job_id"]).json()["status"], "failed")
        self.assertEqual(self.client.get("/api/jobs/job_missing").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/not-valid").status_code, 404)

    def test_initial_job_write_failure_returns_503_without_orphan(self):
        with patch.object(api, "write_json", side_effect=OSError("Disco indisponível ao criar tarefa.")), \
             patch.object(api, "run_search", return_value={"status": "ok"}) as search:
            response = self.client.post("/api/search", json={
                "query": "Consulta", "profile_id": "pf_test",
            })
        self.assertEqual(response.status_code, 503, response.text)
        self.assertTrue(response.json().get("detail"))
        self.assertEqual(api.jobs, {})
        self.assertEqual(list((self.project.root/"jobs").glob("*.json")), [])
        self.assertEqual(self.worker_errors, [])
        search.assert_not_called()

    def test_job_disk_failure_exposes_terminal_state_from_memory(self):
        states = []
        def fail_after_initial_receipt(path, value):
            states.append(value["status"])
            if len(states) == 1:
                return write_json(path, value)
            raise OSError("Disco indisponível durante a tarefa.")
        with patch.object(api, "write_json", side_effect=fail_after_initial_receipt), \
             patch.object(api, "run_search", return_value={"status": "ok"}) as search:
            job = self.wait_for_job(self.client.post("/api/search", json={
                "query": "Consulta", "profile_id": "pf_test",
            }))
            self.worker.shutdown(wait=True)
            current = self.client.get("/api/jobs/"+job["job_id"]).json()
        self.assertEqual(states, ["queued", "running", "failed"])
        self.assertEqual(current["status"], "failed")
        self.assertIn("finished_at", current)
        self.assertIn("Disco indisponível", current["error"])
        self.assertIn("Disco indisponível", current["persistence_warning"])
        self.assertIn(job["job_id"], api.jobs)
        self.assertEqual(read_json(self.project.root/"jobs"/(job["job_id"]+".json"))["status"], "queued")
        self.assertEqual(self.worker_errors, [])
        search.assert_not_called()

    def test_atomic_json_write_retries_transient_windows_permission_error(self):
        destination = self.project.data/"retry.json"
        write_json(destination, {"value": "anterior"})
        real_replace = core_module.os.replace
        attempts = []
        def temporary_lock(source, target):
            attempts.append((source, target))
            if len(attempts) == 1:
                raise PermissionError("Arquivo temporariamente aberto por outro processo.")
            return real_replace(source, target)
        with patch("autorag.core.os.replace", side_effect=temporary_lock), \
             patch("autorag.core.time.sleep") as sleep:
            write_json(destination, {"value": "conteúdo atualizado", "count": 2})
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(0.025)
        self.assertEqual(read_json(destination), {"value": "conteúdo atualizado", "count": 2})
        self.assertEqual(list(self.project.data.glob(".gravar_*")), [])

    def test_queue_limit_rejects_new_work(self):
        with patch.object(api, "MAX_JOBS", 0):
            response = self.client.post("/api/search", json={"query": "Consulta", "profile_id": "pf_test"})
        self.assertEqual(response.status_code, 429)

    def test_restart_marks_interrupted_jobs_failed(self):
        job_id = "job_test_interrupted"
        path = self.project.root/"jobs"/(job_id+".json")
        write_json(path, {"job_id": job_id, "status": "running", "kind": "search"})
        async def restart():
            async with api.lifespan(api.app):
                pass
        asyncio.run(restart())
        result = self.client.get("/api/jobs/"+job_id).json()
        self.assertEqual(result["status"], "failed")
        self.assertIn("reinício", result["error"])

    def test_history_and_fetch_execution_from_isolated_project(self):
        data = {"execution_id": "web_test_saved", "query": "Compras públicas", "date": "2026-10-16",
                "mode": "semantic", "status": "ok", "semantic_results": [], "baseline_results": []}
        write_json(self.project.root/"estudos"/"web"/data["execution_id"]/"resultado.json", data)
        legacy = {"execucao_id": "exp_test_saved", "texto_original_entrada": "Busca legada",
                  "data": "2026-10-15", "status": "concluido"}
        write_json(self.project.root/"estudos"/"execucoes"/legacy["execucao_id"]/"resultado.json", legacy)
        history = self.client.get("/api/experiments").json()["experiments"]
        self.assertEqual(history[0]["execution_id"], data["execution_id"])
        self.assertEqual(history[1]["execution_id"], legacy["execucao_id"])
        self.assertEqual(self.client.get("/api/experiments/"+legacy["execucao_id"]).json(), legacy)
        self.assertEqual(self.client.get("/api/experiments/"+data["execution_id"]).json(), data)
        self.assertEqual(self.client.get("/api/experiments/web_missing").status_code, 404)
        self.assertEqual(self.client.get("/api/experiments/not_an_execution").status_code, 404)

    def test_ingested_source_url_is_presented_for_citations(self):
        topic = {"topico_id": "t1", "titulo": "Art. 1", "conteudo_fonte": "Regra.", "resumo": "Regra."}
        document = {"nome_documento": "Portaria", "fonte": {"referencia": "https://example.org/portaria.pdf"}}
        rows = present_results([{"topic_id": "t1", "distance": 0.2}], {"t1": (topic, document)})
        self.assertEqual(rows[0]["source"]["url"], "https://example.org/portaria.pdf")
        self.assertEqual(rows[0]["text"], "Regra.")


if __name__ == "__main__":
    unittest.main()
