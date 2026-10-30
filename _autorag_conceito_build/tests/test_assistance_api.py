"""Authenticated end-to-end API checks, with no external inference."""
from unittest.mock import patch

import test_comparison_api as helpers


@helpers.unittest.skipUnless(helpers.WEB_AVAILABLE, "Instale requirements-web.txt.")
class AssistanceAPITests(helpers.unittest.TestCase):
    setUp = helpers.ComparisonAPITests.setUp
    tearDown = helpers.ComparisonAPITests.tearDown
    wait_job = helpers.ComparisonAPITests.wait_job

    def test_noise_search_never_calls_any_llm(self):
        with patch("autorag.assistance.make_structured_generator", side_effect=AssertionError("No support call")), \
                patch("autorag.evaluation.make_evaluator", side_effect=AssertionError("No evaluation call")):
            result = self.wait_job(self.client.post("/api/search", json={"query":"Consulta normativa", "noise_filter":{"enabled":True,"margin":2}}))
        self.assertEqual(result["support"]["status"], "noise")
        self.assertEqual(result["analysis"]["status"], "skipped_noise")
        self.assertTrue(all(not e["results"] for e in result["experiments"]))

    def test_laboratory_noise_is_classified_without_support_enabled(self):
        with patch("autorag.assistance.make_structured_generator", side_effect=AssertionError("No LLM")):
            result = self.wait_job(self.client.post("/api/comparisons", json={"query":"Consulta normativa", "noise_filter":{"enabled":True,"margin":2}, "experiments":[{"name":"Laboratório", "profile_ids":["pf_test"], "categories":["conteudo_fonte"]}]}))
        self.assertEqual(result["support"]["status"], "noise")
        self.assertIn("ruído no contexto da base", result["support"]["answer"])

    def test_consultation_answers_from_retrieved_topics(self):
        def answer(data):
            topic_id = data["topics"][0]["topic_id"]
            return {"answer":"Resposta ["+topic_id+"]", "topic_ids":[topic_id]}
        with patch("autorag.assistance.make_structured_generator", return_value=answer):
            result = self.wait_job(self.client.post("/api/search", json={"query":"Consulta normativa"}))
        self.assertEqual(result["support"]["status"], "completed")
        self.assertEqual(result["support"]["provenance"]["model"], "gpt-6-luna")
        saved = self.client.get("/api/comparisons/"+result["comparison_id"]).json()
        self.assertEqual(saved["support"], result["support"])

    def test_settings_are_persistent_and_validate_noise(self):
        response = self.client.put("/api/consultation-config", json={"support":{"enabled":False}, "noise_filter":{"enabled":True,"text":"ruído", "margin":.15}})
        self.assertEqual(response.status_code, 200, response.text)
        current = self.client.get("/api/consultation-config").json()
        self.assertEqual(current["noise_filter"]["margin"], .15)
        self.assertFalse(current["support"]["enabled"])
        bad = self.client.post("/api/search", json={"query":"Teste", "noise_filter":{"margin":-1}})
        self.assertEqual(bad.status_code, 422)

    def test_enrichment_job_activation_and_opt_in_query(self):
        record = self.wait_job(self.client.post("/api/enrichments", json={"text":"Consulta útil", "profile_ids":["pf_test"], "topic_ids":["t1"], "mode":"manual"}))
        self.assertEqual(record["status"], "completed")
        rows = self.client.get("/api/enrichments").json()["entries"]
        self.assertEqual(len(rows), 1)
        self.assertNotIn("vector", rows[0])
        self.assertFalse(rows[0]["active"])
        result = self.wait_job(self.client.patch("/api/enrichments/activation", json={"entry_ids":record["entry_ids"], "active":True}))
        self.assertEqual(result["updated"], 1)
        comparison = self.wait_job(self.client.post("/api/comparisons", json={"query":"Consulta útil", "experiments":[{"name":"Enriquecido", "profile_ids":["pf_test"], "categories":["enriquecimento_recuperacao"]}]}))
        self.assertEqual(comparison["experiments"][0]["results"][0]["topic_id"], "t1")
        self.assertFalse(comparison.get("support", {}).get("answer"))
        runs = self.client.get("/api/enrichment-runs").json()["runs"]
        self.assertEqual(runs[0]["run_id"], record["run_id"])
        choices = self.client.get("/api/enrichment/topics?query=t1").json()
        self.assertEqual(choices["topics"][0]["topic_id"], "t1")

    def test_enrichment_routes_require_authentication(self):
        self.client.headers.pop("Authorization")
        for path in ("/api/enrichments", "/api/enrichment/topics", "/api/enrichment-runs"):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.patch("/api/enrichments/activation", json={"entry_ids":["unknown"],"active":True}).status_code, 401)

    def test_explicit_profile_search_supports_noise_gate(self):
        with patch("autorag.assistance.make_structured_generator", side_effect=AssertionError("No LLM")):
            result = self.wait_job(self.client.post("/api/search", json={"query":"Consulta normativa", "profile_id":"pf_test", "noise_filter":{"enabled":True,"margin":2},"support":{"enabled":True}}))
        self.assertEqual(result["support"]["status"], "noise")
        self.assertEqual(result["semantic_results"], [])

    def test_new_frontend_assets_and_controls_are_served(self):
        html = self.client.get("/").text
        self.assertIn('id="enrichment-form"', html)
        self.assertIn('id="support-answer"', html)
        self.assertEqual(self.client.get("/assistance.js").status_code, 200)
