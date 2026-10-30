"""Offline regression tests for independent model spaces and reproducible comparisons."""
from copy import deepcopy
import csv
import io
import json
import hashlib
import zipfile
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from autorag.core import Project, read_json, write_json
from autorag.comparisons import (active_experiments, consultation_config, export_comparison,
    list_comparisons, list_presets, load_comparison, load_preset, run_comparison,
    save_comparison, save_consultation_config, save_preset, validate_config, reproduce_comparison,
    export_vector_bundle)
from autorag.semantic import search_semantic, vector_search
from autorag.service import generate, import_base
from autorag.web_service import run_search
from test_autorag import FakeProvider, fixture


class Float32QueryProvider(FakeProvider):
    def embed(self, texts, role="documento"):
        if role == "consulta":
            return np.tile(np.asarray([.123, .222, .456], dtype=np.float32), (len(texts), 1))
        return super().embed(texts, role).astype(np.float32)


class FourDimensionalProvider(FakeProvider):
    def embed(self, texts, role="documento"):
        return np.column_stack([super().embed(texts, role), np.ones(len(texts))])


def one_cluster(matrix, params):
    return np.zeros((len(matrix), params["n_components"])), SimpleNamespace(
        labels_=np.zeros(len(matrix), dtype=int), probabilities_=np.ones(len(matrix)))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = Project(self.root/"project")
        source = self.root/"base.json"
        write_json(source, fixture())
        import_base(self.project, source)
        self.profile = {"perfil_id": "pf_test", "provedor": "teste", "modelo": "offline",
                        "dimensoes": 3, "revisao_modelo": "fixed1", "limite_tokens": 512,
                        "normalizacao": "l2", "versao_preparacao": "1",
                        "endpoint": "http://secret.example/?token=DO_NOT_EXPORT", "caminho_local": "/private/model"}
        self.other = {**self.profile, "perfil_id": "pf_other", "modelo": "another", "dimensoes": 4,
                      "revisao_modelo": "fixed2"}
        self.missing = {**self.profile, "perfil_id": "pf_missing"}
        self.project.save_config({"perfis": {p["perfil_id"]: p for p in (self.profile, self.other, self.missing)},
                                  "perfil_ativo_id": "pf_test"})
        self.provider = FakeProvider(self.profile)
        self.other_provider = FourDimensionalProvider(self.other)
        self.providers = {"pf_test": self.provider, "pf_other": self.other_provider}
        generate(self.project, "pf_test", ["conteudo_fonte", "resumo", "palavras_chave"], provider=self.provider)
        generate(self.project, "pf_other", ["conteudo_fonte", "resumo"], provider=self.other_provider)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, name="Teste", **values):
        return {"name": name, "mode": "vector", "profile_ids": ["pf_test"],
                "categories": ["conteudo_fonte"], "top_k": 2, **values}

    def run_configs(self, configs, **request):
        return run_comparison(self.project, {"query": "Consulta normativa", "experiments": configs, **request}, self.providers)

    def test_default_query_uses_only_original_content_without_clustering(self):
        with patch("autorag.semantic._fit_clusters", side_effect=AssertionError("Must not cluster")):
            comparison = run_comparison(self.project, {"query": "Consulta normativa"}, self.providers)
            result = run_search(self.project, {"profile_id": "pf_test", "query": "Outra consulta"}, provider=self.provider)
        experiment = comparison["experiments"][0]
        self.assertEqual(experiment["config"]["mode"], "vector")
        self.assertEqual(experiment["config"]["categories"], ["conteudo_fonte"])
        self.assertEqual(result["mode"], "vector")
        self.assertTrue(all(row["categories"] == ["conteudo_fonte"] for row in result["semantic_results"]))
        self.assertEqual(comparison["analysis"]["status"], "disabled")

    def test_three_distinct_strategies_share_exact_base_snapshot(self):
        configs = [self.config("Distância"), self.config("Agrupamento", mode="semantic", categories=["resumo", "palavras_chave"], min_cluster_size=2),
                   self.config("Média", mode="laboratory", metric="manhattan", categories=["resumo", "conteudo_fonte"])]
        with patch("autorag.semantic._fit_clusters", side_effect=one_cluster):
            result = self.run_configs(configs)
        self.assertEqual(len(result["experiments"]), 3)
        self.assertTrue(all(experiment["status"] == "ok" for experiment in result["experiments"]))
        self.assertTrue(result["base_checksum"])
        self.assertEqual(result["software"]["autorag"], "0.4.0")
        self.assertEqual(load_comparison(self.project, result["comparison_id"], verify=True), result)
        for experiment in result["experiments"]:
            self.assertTrue(experiment["spaces"][0]["indices"]["conteudo_fonte"]["checksum"])
        self.assertEqual(list_comparisons(self.project)[0]["comparison_id"], result["comparison_id"])

    def test_multiple_models_use_rrf_and_keep_distances_in_their_own_spaces(self):
        result = self.run_configs([self.config(profile_ids=["pf_test", "pf_other"], categories=["resumo", "conteudo_fonte"])])
        experiment = result["experiments"][0]
        self.assertEqual(experiment["fusion"]["method"], "rrf")
        self.assertEqual([space["profile"]["dimensoes"] for space in experiment["spaces"]], [3, 4])
        self.assertEqual(len(experiment["results"]), 2)
        for row in experiment["results"]:
            self.assertIsNone(row["distance"])
            self.assertIsNone(row["cosine_similarity"])
            self.assertEqual({proof["profile_id"] for proof in row["model_evidence"]}, {"pf_test", "pf_other"})
            self.assertAlmostEqual(row["fusion_score"], sum(1/(60+proof["rank"]) for proof in row["model_evidence"]))
        self.assertIn("antes da fusão", experiment["warnings"][-1])
        first = self.run_configs([self.config(profile_ids=["pf_test", "pf_other"], categories=["resumo", "conteudo_fonte"])])
        self.assertEqual(first["experiments"][0]["results"], experiment["results"])

    def test_missing_model_index_fails_one_experiment_without_partial_fusion(self):
        result = self.run_configs([self.config("Disponível"), self.config("Índice ausente", profile_ids=["pf_test", "pf_missing"])])
        valid, invalid = result["experiments"]
        self.assertEqual(valid["status"], "ok")
        self.assertTrue(valid["results"])
        self.assertEqual(invalid["status"], "error")
        self.assertFalse(invalid["results"])
        self.assertEqual(invalid["profile_errors"][0]["profile_id"], "pf_missing")
        self.assertEqual(result["status"], "completed")

    def test_distinct_categories_per_model_use_only_the_selected_spaces_and_reproduce(self):
        self.project.path("indices/pf_test/resumo/atual.json").unlink()
        config = self.config("Combinação livre", profile_ids=["pf_test", "pf_other"],
                             model_categories={"pf_test": ["palavras_chave"], "pf_other": ["resumo"]})
        result = self.run_configs([config])
        experiment = result["experiments"][0]
        self.assertEqual(experiment["status"], "ok")
        self.assertEqual(experiment["config"]["categories"], ["palavras_chave", "resumo"])
        self.assertEqual([space["categories"] for space in experiment["spaces"]], [["palavras_chave"], ["resumo"]])
        for row in experiment["results"]:
            for proof in row["model_evidence"]:
                expected = config["model_categories"][proof["profile_id"]]
                self.assertEqual(proof["categories"], expected)
                self.assertTrue(all(match["category"] in expected for match in proof["matched_vectors"]))
        reproduced = reproduce_comparison(self.project, result["comparison_id"])
        self.assertEqual(reproduced["experiments"], result["experiments"])
        exported = json.loads(export_comparison(self.project, result["comparison_id"], "config"))
        self.assertEqual(exported["experiments"][0]["model_categories"], config["model_categories"])
        with patch("autorag.semantic._fit_clusters", side_effect=one_cluster):
            semantic = self.run_configs([{**config, "mode": "semantic", "min_cluster_size": 2}])
        self.assertEqual(semantic["experiments"][0]["status"], "ok")

    def test_per_model_category_validation_and_missing_index_are_explicit(self):
        for choices in (None, [], {}, {"pf_test": ["resumo"]},
                        {"pf_test": ["resumo"], "pf_other": ["resumo"], "pf_extra": ["resumo"]},
                        {"pf_test": [], "pf_other": ["resumo"]},
                        {"pf_test": "resumo", "pf_other": ["resumo"]},
                        {"pf_test": [{}], "pf_other": ["resumo"]},
                        {"pf_test": ["unknown"], "pf_other": ["resumo"]}):
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                validate_config(self.project, self.config(profile_ids=["pf_test", "pf_other"], model_categories=choices))
        comparison = self.run_configs([self.config(profile_ids=["pf_test", "pf_other"],
            model_categories={"pf_test": ["resumo"], "pf_other": ["palavras_chave"]})])
        failed = comparison["experiments"][0]
        self.assertEqual(failed["status"], "error")
        self.assertEqual([error["profile_id"] for error in failed["profile_errors"]], ["pf_other"])
        self.assertEqual(failed["spaces"][0]["categories"], ["resumo"])
        self.assertTrue(failed["spaces"][0]["baseline_results"])

    def test_vector_bundle_contains_exact_numeric_arrays_metrics_and_deduplicated_rows(self):
        comparison = self.run_configs([self.config(profile_ids=["pf_test", "pf_other"], categories=["resumo", "conteudo_fonte"])],
                                     query="Consulta normativa extensa " * 50)
        data = export_vector_bundle(self.project, comparison["comparison_id"], persist=False)
        self.assertEqual(data, export_vector_bundle(self.project, comparison["comparison_id"]))
        archive = zipfile.ZipFile(io.BytesIO(data))
        manifest = json.loads(archive.read("manifesto.json"))
        self.assertEqual({profile["dimensions"] for profile in manifest["profiles"]}, {3, 4})
        self.assertTrue({"retrieval", "baseline"}.issubset({row["scope"] for row in manifest["associations"]}))
        for entry in manifest["files"]:
            self.assertEqual(hashlib.sha256(archive.read(entry["path"])).hexdigest(), entry["sha256"])
        total_vectors = 0
        for profile in manifest["profiles"]:
            with np.load(io.BytesIO(archive.read(profile["query_file"])), allow_pickle=False) as package:
                query = package["vetores"]
                query_effective = package["vetores_efetivos"][0]
                self.assertEqual(query.shape, (1, profile["dimensions"]))
                with np.load(self.project.path(profile["query_reference"]["arquivo_npz"]), allow_pickle=False) as source:
                    self.assertEqual(query.dtype, source["vetores"].dtype)
                    np.testing.assert_array_equal(query, source["vetores"])
            with np.load(io.BytesIO(archive.read(profile["query_source_file"])), allow_pickle=False) as package:
                self.assertGreater(len(package["vetores_segmentos"]), 1)
                with np.load(self.project.path(profile["query_source_reference"]["arquivo_npz"]), allow_pickle=False) as source:
                    for name in package.files:
                        np.testing.assert_array_equal(package[name], source[name])
            with np.load(io.BytesIO(archive.read(profile["vectors_file"])), allow_pickle=False) as package:
                vectors, effective = package["vetores"], package["vetores_efetivos"]
                self.assertEqual(vectors.shape, (len(profile["entries"]), profile["dimensions"]))
                self.assertTrue(all(package[name].dtype.kind in "fiu" for name in package.files))
                total_vectors += len(vectors)
                for entry in profile["entries"]:
                    reference = entry["source_reference"]
                    with np.load(self.project.path(reference["arquivo_npz"]), allow_pickle=False) as source:
                        np.testing.assert_array_equal(vectors[entry["matrix_row"]], source["vetores"][reference["linha"]])
                for row in manifest["associations"]:
                    if row["profile_id"] != profile["profile_id"]:
                        continue
                    vector = effective[row["matrix_row"]]
                    self.assertAlmostEqual(row["vector_distance"], 1-float(vector @ query_effective), places=12)
                    self.assertIsNotNone(row["fusion_contribution"])
        self.assertLess(total_vectors, len(manifest["associations"]))
        self.assertTrue(list(csv.DictReader(io.StringIO(archive.read("metricas.csv").decode("utf-8-sig")))))
        reproduced = reproduce_comparison(self.project, comparison["comparison_id"])
        again = export_vector_bundle(self.project, reproduced["comparison_id"], persist=False)
        with zipfile.ZipFile(io.BytesIO(again)) as replay_archive:
            replay_manifest = json.loads(replay_archive.read("manifesto.json"))
            for profile in replay_manifest["profiles"]:
                with np.load(io.BytesIO(replay_archive.read(profile["vectors_file"])), allow_pickle=False) as after:
                    with np.load(io.BytesIO(archive.read(profile["vectors_file"])), allow_pickle=False) as before:
                        np.testing.assert_array_equal(after["vetores"], before["vetores"])

    def test_cached_vector_bundle_revalidates_artifacts_and_final_analysis_is_exported_safely(self):
        comparison = self.run_configs([self.config()], query="=Consulta perigosa")
        first = export_vector_bundle(self.project, comparison["comparison_id"])
        row = comparison["experiments"][0]["results"][0]
        row["title"] = "@Título"
        row["analysis"] = {"status": "succeeded", "classification": "pertinente", "relevant": True,
                           "comment": '=HYPERLINK("malicious")', "utility_rank": 1}
        save_comparison(self.project, comparison)
        final = export_vector_bundle(self.project, comparison["comparison_id"], persist=False)
        self.assertNotEqual(first, final)
        with zipfile.ZipFile(io.BytesIO(final)) as archive:
            records = list(csv.DictReader(io.StringIO(archive.read("metricas.csv").decode("utf-8-sig"))))
            self.assertTrue(any(record["comment"].startswith("'=HYPERLINK") for record in records))
            self.assertTrue(any(record["title"] == "'@Título" for record in records))
            manifest = json.loads(archive.read("manifesto.json"))
            self.assertTrue(any((record.get("analysis") or {}).get("comment", "").startswith("=HYPERLINK") for record in manifest["associations"]))
        existing_csv = list(csv.DictReader(io.StringIO(export_comparison(self.project, comparison["comparison_id"], "csv"))))
        self.assertEqual(existing_csv[0]["query"], "'=Consulta perigosa")
        self.assertTrue(existing_csv[0]["comment"].startswith("'=HYPERLINK"))
        space = comparison["experiments"][0]["spaces"][0]
        source_path = self.project.path(space["input_source_reference"]["arquivo_npz"])
        source_path.write_bytes(source_path.read_bytes()+b"modified")
        with self.assertRaisesRegex(ValueError, "segmentos"):
            export_vector_bundle(self.project, comparison["comparison_id"], persist=False)

    def test_missing_category_keeps_original_content_baseline_per_model(self):
        result = self.run_configs([self.config("Falta palavras-chave", profile_ids=["pf_test", "pf_other"], categories=["palavras_chave"])])
        experiment = result["experiments"][0]
        self.assertEqual(experiment["status"], "error")
        self.assertFalse(experiment["results"])
        self.assertEqual(len(experiment["spaces"]), 2)
        self.assertTrue(all(space["baseline_results"] for space in experiment["spaces"]))
        self.assertTrue(experiment["baseline_results"])
        self.assertTrue(export_comparison(self.project, result["comparison_id"], "json"))

    def test_missing_profile_index_error_can_be_exported(self):
        result = self.run_configs([self.config(profile_ids=["pf_test", "pf_missing"])])
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["experiments"][0]["spaces"][0]["baseline_results"])
        self.assertFalse(result["experiments"][0]["spaces"][1]["baseline_results"])
        self.assertEqual(json.loads(export_comparison(self.project, result["comparison_id"]))["status"], "error")

    def test_preset_and_active_consultation_settings_are_reusable(self):
        record = save_preset(self.project, {"name": "UMAP com palavras chave", "experiments": [self.config()], "active": True})
        self.assertEqual(load_preset(self.project, record["id"]), record)
        self.assertEqual(list_presets(self.project), [record])
        self.assertEqual(active_experiments(self.project), record["experiments"])
        result = run_comparison(self.project, {"query": "Consulta normativa", "preset_id": record["id"]}, self.providers)
        self.assertEqual(result["experiments"][0]["config"], record["experiments"][0])
        settings = save_consultation_config(self.project, {"experiments": [], "analysis": {"enabled": False}})
        self.assertFalse(active_experiments(self.project))
        self.assertEqual(consultation_config(self.project), settings)

    def test_jsonl_csv_and_config_preserve_evaluation_and_provenance_without_credentials(self):
        comparison = self.run_configs([self.config()], query="=Consulta normativa")
        assessment = {"status": "succeeded", "classification": "pertinente", "relevant": True,
                      "comment": "O conteúdo responde à consulta.", "utility_rank": 1}
        comparison["experiments"][0]["results"][0]["analysis"] = assessment
        save_comparison(self.project, comparison)
        lines = [json.loads(line) for line in export_comparison(self.project, comparison["comparison_id"], "jsonl").splitlines()]
        self.assertEqual(lines[0]["analysis"], assessment)
        self.assertTrue(lines[0]["provenance"][0]["profile_fingerprint"])
        disk = (self.project.root/"estudos"/"comparacoes"/comparison["comparison_id"]/"linhas.jsonl").read_text(encoding="utf-8")
        self.assertEqual([json.loads(line) for line in disk.splitlines()], lines)
        table = list(csv.DictReader(io.StringIO(export_comparison(self.project, comparison["comparison_id"], "csv"))))
        self.assertEqual(table[0]["classification"], "pertinente")
        self.assertEqual(table[0]["query"], "'=Consulta normativa")
        for format in ("json", "jsonl", "csv", "config"):
            exported = export_comparison(self.project, comparison["comparison_id"], format)
            self.assertNotIn("DO_NOT_EXPORT", exported)
            self.assertNotIn("/private/model", exported)
        config = json.loads(export_comparison(self.project, comparison["comparison_id"], "config"))
        self.assertEqual(config["experiments"][0], comparison["experiments"][0]["config"])

    def test_later_source_and_index_updates_do_not_change_captured_experiments(self):
        from autorag import comparisons
        original = comparisons._space
        calls = []
        before = self.project.current("pf_test", "resumo")
        def mutate_after_capture(snapshot, config, mapping):
            if not calls:
                base = self.project.base()
                base["documentos"][0]["estrutura"][0]["conteudo_fonte"] = "Texto publicado depois da captura"
                base["documentos"][0]["estrutura"][0]["resumo"] = "Outro resumo"
                write_json(self.project.base_path, base)
                generate(self.project, "pf_test", ["resumo"], provider=self.provider)
            calls.append(True)
            return original(snapshot, config, mapping)
        with patch("autorag.comparisons._space", side_effect=mutate_after_capture):
            comparison = self.run_configs([self.config("Primeiro", categories=["resumo"]), self.config("Segundo", categories=["resumo"])])
        self.assertNotEqual(before, self.project.current("pf_test", "resumo"))
        for experiment in comparison["experiments"]:
            self.assertEqual(experiment["spaces"][0]["indices"]["resumo"]["manifesto"], before["manifesto"])
            self.assertFalse(any(row["text"] == "Texto publicado depois da captura" for row in experiment["results"]))
        load_comparison(self.project, comparison["comparison_id"], verify=True)

    def test_exact_reproduction_uses_saved_vectors_after_updates_without_inference(self):
        original = self.run_configs([self.config("Múltiplos", profile_ids=["pf_test", "pf_other"], categories=["resumo"])])
        base = self.project.base()
        base["documentos"][0]["estrutura"][0]["resumo"] = "Novo conteúdo no índice corrente"
        base["documentos"][0]["estrutura"][0]["conteudo_fonte"] = "Nova fonte que não deve aparecer na reprodução"
        write_json(self.project.base_path, base)
        generate(self.project, "pf_test", ["resumo"], provider=self.provider)
        with patch("autorag.comparisons.save_input", side_effect=AssertionError("Inference forbidden")):
            reproduced = reproduce_comparison(self.project, original["comparison_id"])
        self.assertEqual(reproduced["reproduced_from"], original["comparison_id"])
        self.assertNotEqual(reproduced["comparison_id"], original["comparison_id"])
        self.assertEqual(reproduced["base_checksum"], original["base_checksum"])
        self.assertEqual(reproduced["experiments"], original["experiments"])
        self.assertEqual(reproduced["analysis"]["status"], "disabled")
        load_comparison(self.project, reproduced["comparison_id"], verify=True)

    def test_float32_cached_query_is_stored_exactly_once_per_model_and_reproduced(self):
        from autorag import comparisons
        provider = Float32QueryProvider(self.profile)
        query = "Vetor float32 não trivial"
        run_comparison(self.project, {"query": query, "experiments": [self.config()]}, {"pf_test": provider})
        actual, original_save = [], comparisons.save_input
        def record(*args, **kwargs):
            result = original_save(*args, **kwargs)
            actual.append(result[2].copy())
            return result
        with patch("autorag.comparisons.save_input", side_effect=record):
            saved = run_comparison(self.project, {"query": query, "experiments": [
                self.config("Resumo", categories=["resumo"]), self.config("Fonte")]}, {"pf_test": provider})
        self.assertEqual(len(actual), 1)
        space = saved["experiments"][0]["spaces"][0]
        self.assertTrue(space["input_source_reference"])
        with np.load(self.project.path(space["input_reference"]["arquivo_npz"]), allow_pickle=False) as package:
            np.testing.assert_array_equal(package["vetores"][0], actual[0])
        self.assertEqual(saved["experiments"][1]["spaces"][0]["input_reference"], space["input_reference"])
        reproduced = reproduce_comparison(self.project, saved["comparison_id"])
        self.assertEqual(saved["experiments"], reproduced["experiments"])

    def test_clustering_failure_keeps_baseline_and_does_not_block_other_experiments(self):
        with patch("autorag.semantic._fit_clusters", side_effect=RuntimeError("clustering failed")):
            comparison = self.run_configs([self.config("UMAP", mode="semantic", categories=["resumo", "conteudo_fonte"]),
                                           self.config("Direto")])
        semantic, direct = comparison["experiments"]
        self.assertEqual(semantic["status"], "error")
        self.assertFalse(semantic["results"])
        self.assertTrue(semantic["baseline_results"])
        self.assertTrue(semantic["spaces"][0]["baseline_results"])
        self.assertEqual(direct["status"], "ok")
        self.assertTrue(direct["results"])
        self.assertEqual(load_comparison(self.project, comparison["comparison_id"], verify=True), comparison)

    def test_one_model_clustering_failure_never_becomes_partial_rrf_success(self):
        def fail_one_space(matrix, params):
            if matrix.shape[1] == 4:
                raise RuntimeError("one model clustering failed")
            return one_cluster(matrix, params)
        with patch("autorag.semantic._fit_clusters", side_effect=fail_one_space):
            result = self.run_configs([self.config(mode="semantic", profile_ids=["pf_test", "pf_other"],
                                                   categories=["resumo", "conteudo_fonte"])])
        experiment = result["experiments"][0]
        self.assertEqual(experiment["status"], "error")
        self.assertTrue(experiment["spaces"][0]["results"])
        self.assertFalse(experiment["spaces"][1]["results"])
        self.assertFalse(experiment["results"])
        self.assertTrue(experiment["baseline_results"])
        self.assertEqual(experiment["profile_errors"][0]["profile_id"], "pf_other")

    def test_export_detects_changed_artifacts(self):
        comparison = self.run_configs([self.config()])
        reference = comparison["experiments"][0]["spaces"][0]["indices"]["conteudo_fonte"]
        manifest = self.project.path(reference["manifesto"])
        manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifesto"):
            export_comparison(self.project, comparison["comparison_id"])

    def test_mismatched_profile_index_does_not_return_results(self):
        pointer = self.project.current("pf_test", "conteudo_fonte")
        path = self.project.path(pointer["manifesto"])
        manifest = read_json(path)
        manifest["perfil"]["revisao_modelo"] = "changed"
        write_json(path, manifest)
        result = self.run_configs([self.config()])
        self.assertEqual(result["experiments"][0]["status"], "error")
        self.assertIn("diverge", result["experiments"][0]["profile_errors"][0]["error"])

    def test_invalid_parameters_and_more_than_three_experiments_are_rejected(self):
        for values in ({"mode": []}, {"mode": {}}, {"individual_rule": []}, {"individual_rule": {}},
                       {"categories": [{}]}, {"categories": [[]]}, {"metric": []}, {"metric": {}},
                       {"cluster_metric": []}, {"umap_metric": {}}, {"cluster_selection_method": []}, {"top_k": True}, {"minimum_similarity": 1.1}, {"maximum_distance": float("nan")},
                       {"metric": "invalid"}, {"dendrogram": "yes"}, {"api_key": "secret"},
                       {"profile_ids": ["pf_test", "pf_test"]}, {"n_neighbors": 1}, {"fusion_depth": 101}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_config(self.project, self.config(**values))
        with self.assertRaises(ValueError):
            self.run_configs([self.config(str(i)) for i in range(4)])
        with self.assertRaises(ValueError):
            load_preset(self.project, "../outside")


class ThresholdTests(unittest.TestCase):
    def test_thresholds_are_applied_before_vector_and_semantic_top_k(self):
        def vector(topic, category, similarity):
            return {"topic_id": topic, "category": category, "embedding_id": topic+category,
                    "embedding": [similarity, (1-similarity**2)**0.5]}
        vectors = [vector("many_categories", "resumo", .2), vector("many_categories", "palavras_chave", .3),
                   vector("relevant", "resumo", .9), vector("source", "conteudo_fonte", .85)]
        direct = vector_search(vectors, [1, 0], ["resumo", "palavras_chave"], top_k=1, minimum_similarity=.8)
        self.assertEqual(direct[0]["topic_id"], "relevant")
        with patch("autorag.semantic._fit_clusters", side_effect=one_cluster):
            semantic = search_semantic(vectors, [1, 0], ["resumo", "palavras_chave"], top_k=1,
                                       options={"min_cluster_size": 2}, minimum_similarity=.8)
        self.assertEqual(semantic["semantic_results"][0]["topic_id"], "relevant")
        self.assertEqual(semantic["baseline_results"][0]["topic_id"], "source")
        self.assertEqual(vector_search(vectors, [1, 0], ["resumo"], maximum_distance=.05), [])


if __name__ == "__main__":
    unittest.main()
