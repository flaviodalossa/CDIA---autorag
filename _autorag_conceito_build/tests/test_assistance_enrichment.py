"""Offline acceptance tests for answers, noise rejection and opt-in vector enrichment."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from autorag.assistance import (NOISE_MESSAGE, answer_comparison, capture_noise, filter_rows,
                                noise_config, support_config)
from autorag.comparisons import (run_comparison, reproduce_comparison, save_consultation_config,
                                consultation_config, save_preset)
from autorag.core import Project, digest, normalize, topics, write_json
from autorag.enrichment import (CATEGORY, add_entry, list_entries, run_enrichment, set_active,
                                topic_choices)
from autorag.service import generate, import_base
from test_autorag import FakeProvider, fixture


class InputProvider(FakeProvider):
    def embed(self, texts, role="documento"):
        if role == "consulta":
            self.calls.append((list(texts), role))
            return np.asarray([[0., 1., 0.] if t == "segunda" else [1., 0., 0.] for t in texts])
        values = super().embed(texts, role)
        for i, text in enumerate(texts):
            if text == "referência": values[i] = [1., 0., 0.]
            if text == "segunda": values[i] = [0., 1., 0.]
        return values


class AssistanceEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Project(Path(self.temp.name)/"project")
        source = Path(self.temp.name)/"base.json"
        write_json(source, fixture())
        import_base(self.project, source)
        self.profile = {"perfil_id": "pf_test", "provedor": "teste", "modelo": "offline",
                        "dimensoes": 3, "revisao_modelo": "fixed", "limite_tokens": 512,
                        "normalizacao": "l2", "versao_preparacao": "1"}
        self.project.save_config({"perfis": {"pf_test": self.profile}, "perfil_ativo_id": "pf_test"})
        self.provider = InputProvider(self.profile)
        self.providers = {"pf_test": self.provider}
        generate(self.project, "pf_test", ["conteudo_fonte", "resumo"], provider=self.provider)

    def tearDown(self):
        self.temp.cleanup()

    def compare(self, **extra):
        return run_comparison(self.project, {"query": "primeira", **extra}, self.providers)

    def test_noise_reference_is_retrieved_each_search_without_new_embedding(self):
        config = noise_config({"enabled": True, "text": "referência", "margin": .03})
        with self.project.lock():
            one = capture_noise(self.project, np.array([1., 0., 0.]), self.profile, config, self.provider)
            count = len(self.provider.calls)
            two = capture_noise(self.project, np.array([0., 1., 0.]), self.profile, config, self.provider)
        self.assertEqual(len(self.provider.calls), count)
        self.assertEqual(one["reference"], two["reference"])
        self.assertEqual(one["embedding_role"], "documento")
        self.assertEqual(self.provider.calls[-1][1], "documento")
        self.assertAlmostEqual(one["threshold"], 1.03)
        self.assertAlmostEqual(two["threshold"], .03)

    def test_filter_is_strict_and_uses_original_cosine_for_projected_results(self):
        rows = [{"topic_id": "below", "cosine_similarity": .49},
                {"topic_id": "equal", "cosine_similarity": .5},
                {"topic_id": "above", "cosine_similarity": .5001},
                {"topic_id": "projected", "cosine_similarity": .99, "original_cosine_similarity": .4}]
        self.assertEqual([r["topic_id"] for r in filter_rows(rows, {"threshold": .5})], ["above"])

    def test_all_noise_skips_even_construction_of_llm(self):
        result = self.compare(noise_filter={"enabled": True, "margin": 2})
        self.assertFalse(result["experiments"][0]["results"])
        self.assertFalse(result["experiments"][0]["baseline_results"])
        with patch("autorag.assistance.make_structured_generator", side_effect=AssertionError("Must not call")):
            answer_comparison(result)
        self.assertEqual(result["support"]["status"], "noise")
        self.assertEqual(result["support"]["answer"], NOISE_MESSAGE)

    def test_answer_uses_only_returned_source_topics_and_records_citations(self):
        result = self.compare()
        first = result["experiments"][0]["results"][0]
        generator = Mock(return_value={"answer": "Resposta fundamentada ["+first["topic_id"]+"]", "topic_ids": [first["topic_id"]]})
        answer_comparison(result, generator=generator)
        self.assertEqual(result["support"]["status"], "completed")
        data = generator.call_args.args[0]
        self.assertEqual(data["query"], "primeira")
        self.assertEqual({r["topic_id"] for r in data["topics"]}, {r["topic_id"] for r in result["experiments"][0]["results"]})
        self.assertTrue(all(r["text"].startswith("Fonte completa") for r in data["topics"]))

    def test_invalid_llm_citation_preserves_retrieval(self):
        result = self.compare()
        rows = deepcopy(result["experiments"])
        answer_comparison(result, generator=lambda _: {"answer": "Fabricado", "topic_ids": ["unknown"]})
        self.assertEqual(result["support"]["status"], "failed")
        self.assertEqual(result["experiments"], rows)

    def test_noise_reproduction_uses_saved_reference_without_inference(self):
        original = self.compare(noise_filter={"enabled": True, "text": "segunda", "margin": .01})
        with patch("autorag.experiments.get_provider", side_effect=AssertionError("No inference")):
            repeated = reproduce_comparison(self.project, original["comparison_id"])
        self.assertEqual(original["experiments"], repeated["experiments"])
        self.assertEqual(repeated["noise_filter"], original["noise_filter"])
        self.assertEqual(repeated["support"]["status"], "disabled")

    def test_settings_preserve_support_and_noise_when_lab_applies_experiments(self):
        settings = save_consultation_config(self.project, {"support": {"enabled": False}, "noise_filter": {"enabled": True, "margin": .12}})
        self.assertTrue(settings["noise_filter"]["enabled"])
        saved = save_consultation_config(self.project, {"experiments": [], "analysis": {"enabled": False}})
        self.assertEqual(saved["support"], settings["support"])
        self.assertEqual(saved["noise_filter"], settings["noise_filter"])
        self.assertEqual(consultation_config(self.project), saved)

    def test_activating_preset_preserves_noise_and_support(self):
        original = save_consultation_config(self.project, {"support": {"enabled": False}, "noise_filter": {"enabled": True, "margin": .15}})
        save_preset(self.project, {"name": "Novo", "active": True, "experiments": [{"name": "Padrão", "profile_ids": ["pf_test"], "categories": ["resumo"]}]})
        current = consultation_config(self.project)
        self.assertEqual(current["support"], original["support"])
        self.assertEqual(current["noise_filter"], original["noise_filter"])

    def test_drafts_survive_restart_without_entering_production(self):
        ids = add_entry(self.project, "primeira", ["pf_test"], ["t1", "t2"], providers=self.providers)
        restored = Project(self.project.root)
        self.assertEqual(list_entries(restored)[0]["id"], ids[0])
        self.assertFalse(list_entries(restored)[0]["active"])
        self.assertIsNone(restored.current("pf_test", CATEGORY))
        self.assertNotIn("vector", list_entries(restored)[0])

    def test_activation_publishes_normalized_mean_and_deactivation_retains_inputs(self):
        one = add_entry(self.project, "primeira", ["pf_test"], ["t1"], providers=self.providers)
        two = add_entry(self.project, "segunda", ["pf_test"], ["t1"], providers=self.providers)
        set_active(self.project, one+two, True)
        manifest, matrix, _ = self.project.load_index("pf_test", CATEGORY)
        self.assertEqual(len(manifest["itens"]), 1)
        self.assertEqual(set(manifest["itens"][0]["enrichment_ids"]), set(one+two))
        np.testing.assert_allclose(matrix[0], np.array([1.,1.,0.])/np.sqrt(2))
        snapshot = self.compare(experiments=[{"name":"Enriched", "profile_ids":["pf_test"], "categories":[CATEGORY]}])
        self.assertEqual(snapshot["experiments"][0]["results"][0]["topic_id"], "t1")
        set_active(self.project, one+two, False)
        self.assertEqual(len(self.project.load_index("pf_test", CATEGORY)[1]), 0)
        self.assertEqual(len(list_entries(self.project)), 2)
        repeated = reproduce_comparison(self.project, snapshot["comparison_id"])
        self.assertEqual(snapshot["experiments"], repeated["experiments"])

    def test_manual_enrichment_does_not_call_llm(self):
        with patch("autorag.enrichment._generator", side_effect=AssertionError("No LLM")):
            record = run_enrichment(self.project, {"text":"primeira", "profile_ids":["pf_test"], "topic_ids":["t1"]}, self.providers)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(len(record["entry_ids"]), 1)
        self.assertFalse(list_entries(self.project)[0]["active"])

    def test_automatic_selection_can_refuse_all_candidates(self):
        record = run_enrichment(self.project, {"text":"primeira", "profile_ids":["pf_test"], "mode":"automatic"}, self.providers,
                                selector=lambda data: {"topic_ids":[], "reason":"Não há suporte."})
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["entry_ids"], [])

    def test_loop_runs_exact_user_count_and_saves_each_generated_input(self):
        generator = Mock(side_effect=[{"query":"primeira"},{"query":"segunda"},{"query":"terceira"}])
        selector = Mock(return_value={"topic_ids":["t1"],"reason":"Tópico útil."})
        record = run_enrichment(self.project, {"text":"Pesquisar", "mode":"loop", "iterations":3,
                    "profile_ids":["pf_test"],"topic_ids":["t1","t2"]}, self.providers, selector=selector, query_generator=generator)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(generator.call_count, 3)
        self.assertEqual(selector.call_count, 3)
        self.assertEqual(len(record["iterations"]), 3)
        self.assertEqual(len(list_entries(self.project)), 3)
        self.assertEqual(generator.call_args.args[0]["previous_queries"], ["primeira","segunda"])
        self.assertTrue(all(not r["active"] for r in list_entries(self.project)))

    def test_loop_failure_preserves_completed_iterations_as_drafts(self):
        generator = Mock(side_effect=[{"query":"primeira"}, RuntimeError("provider secret")])
        record = run_enrichment(self.project, {"text":"Pesquisar", "mode":"loop", "iterations":3,
                "profile_ids":["pf_test"],"topic_ids":["t1"]}, self.providers,
                selector=lambda _: {"topic_ids":["t1"],"reason":"Útil"}, query_generator=generator)
        self.assertEqual(record["status"], "partial")
        self.assertEqual(len(record["iterations"]), 1)
        self.assertEqual(len(list_entries(self.project)), 1)
        self.assertNotIn("secret", record["error"])

    def test_llm_cannot_associate_unknown_topics(self):
        result = run_enrichment(self.project, {"text":"primeira", "mode":"automatic", "profile_ids":["pf_test"]}, self.providers,
                                selector=lambda _: {"topic_ids":["inventado"],"reason":"Não permitido"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(list_entries(self.project), [])

    def test_changed_source_prevents_activation_and_removes_existing_vector_from_retrieval(self):
        ids = add_entry(self.project, "primeira", ["pf_test"], ["t1"], providers=self.providers)
        set_active(self.project, ids, True)
        base = self.project.base()
        base["documentos"][0]["estrutura"][0]["conteudo_fonte"] += " modificado"
        write_json(self.project.base_path, base)
        self.assertEqual(list_entries(self.project)[0]["stale_topic_ids"], ["t1"])
        with self.assertRaises(ValueError):
            set_active(self.project, ids, True)
        result = self.compare(experiments=[{"name":"Enriched", "profile_ids":["pf_test"], "categories":[CATEGORY]}])
        self.assertFalse(result["experiments"][0]["results"])
        set_active(self.project, ids, False)

    def test_publication_failure_rolls_back_all_current_pointers(self):
        ids = add_entry(self.project, "primeira", ["pf_test"], ["t1"], providers=self.providers)
        with patch.object(self.project, "publish", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                set_active(self.project, ids, True)
        self.assertFalse(list_entries(self.project)[0]["active"])
        self.assertIsNone(self.project.current("pf_test", CATEGORY))

    def test_text_deduplication_and_topic_pagination(self):
        one = add_entry(self.project, "primeira", ["pf_test"], ["t1"], providers=self.providers)
        two = add_entry(self.project, "primeira", ["pf_test"], ["t1"], providers=self.providers)
        self.assertEqual(one, two)
        choices = topic_choices(self.project, "Fonte completa", 1, 1)
        self.assertEqual(choices["total"], 3)
        self.assertEqual(choices["topics"][0]["topic_id"], "t2")

    def test_invalid_configuration_is_rejected(self):
        for value in ({"margin":float("nan")},{"enabled":"yes"},{"text":" "},{"margin":-.1}):
            with self.assertRaises(ValueError): noise_config(value)
        with self.assertRaises(ValueError): support_config({"enabled":True,"model":" "})
        with self.assertRaises(ValueError): generate(self.project, "pf_test", [CATEGORY], provider=self.provider)


if __name__ == "__main__":
    unittest.main()
