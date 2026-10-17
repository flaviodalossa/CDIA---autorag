"""Behavioral tests for independent reduction/clustering and auditable exports."""
import csv
import hashlib
import importlib.util
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import numpy as np

from autorag import semantic
from autorag.comparisons import (export_comparison, export_vector_bundle,
    reproduce_comparison, save_preset, validate_config)
import test_comparisons as fixtures
from test_semantic import fake_model, row


def projection(matrix, params):
    # Deliberately not centered or normalized: L2 projected rankings differ.
    count = len(matrix)
    result = np.zeros((count, params["n_components"]), dtype=float)
    result[:, 0] = np.arange(count, 0, -1) * 2
    result[:, 1] = 3
    return result


class IndependentStageTests(unittest.TestCase):
    def setUp(self):
        self.vectors = [row(str(i), "conteudo_fonte", .1+i*.08) for i in range(7)]

    def test_umap_only_ranks_raw_projection_and_never_clusters(self):
        with patch.object(semantic, "_fit_umap", side_effect=projection), \
                patch.object(semantic, "_fit_hdbscan", side_effect=AssertionError("must not cluster")):
            result = semantic.search_semantic(self.vectors, [1, 0], options={
                "use_hdbscan": False, "metric": "euclidean", "n_components": 2}, top_k=3)
        self.assertEqual([r["topic_id"] for r in result["semantic_results"]], ["6", "5", "4"])
        self.assertEqual([r["distance"] for r in result["semantic_results"]], [2, 4, 6])
        self.assertEqual(result["parameters"]["distance_space"], "umap_projection")
        self.assertIsNone(result["cluster"]["space"])
        self.assertEqual(result["cluster"]["points"], [])
        self.assertIsNone(result["dendrogram"])
        self.assertEqual(result["projection"]["normalization"], "none")
        self.assertTrue(all(r["distance_space"] == "original_l2_normalized" for r in result["baseline_results"]))
        best = result["semantic_results"][0]
        self.assertAlmostEqual(best["original_distance"], np.linalg.norm(np.asarray(self.vectors[6]["embedding"])-[1, 0]))
        self.assertNotEqual(best["original_distance"], best["projected_distance"])

    def test_umap_threshold_distance_is_projected_similarity_is_original(self):
        with patch.object(semantic, "_fit_umap", side_effect=projection):
            result = semantic.search_semantic(self.vectors, [1, 0], options={
                "use_hdbscan": False, "metric": "euclidean", "n_components": 2}, maximum_distance=4)
            filtered = semantic.search_semantic(self.vectors, [1, 0], options={
                "use_hdbscan": False, "metric": "euclidean", "n_components": 2},
                maximum_distance=4, minimum_similarity=.99)
        self.assertEqual([r["topic_id"] for r in result["semantic_results"]], ["6", "5"])
        self.assertEqual(filtered["semantic_results"], [])
        self.assertTrue(filtered["baseline_results"])

    def test_hdbscan_only_uses_original_coordinates_without_umap_size_constraints(self):
        with patch.object(semantic, "_fit_umap", side_effect=AssertionError("must not reduce")), \
                patch.object(semantic, "_fit_hdbscan", return_value=fake_model([0, 0, 0])) as fit:
            result = semantic.search_semantic(self.vectors[:2], [2, 0], options={
                "use_umap": False, "min_cluster_size": 2, "min_samples": 1, "dendrogram": True})
        np.testing.assert_allclose(fit.call_args.args[0][-1], [1, 0])
        np.testing.assert_allclose(np.linalg.norm(fit.call_args.args[0], axis=1), 1)
        self.assertIsNone(result["projection"])
        self.assertEqual(result["cluster"]["space"], "original_l2_normalized")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["dendrogram"]["point_labels"][-1], "__query__")
        self.assertNotIn("effective_n_components", result["parameters"])

    def test_neither_stage_is_direct_vector_retrieval(self):
        with patch.object(semantic, "_fit_umap", side_effect=AssertionError()), \
                patch.object(semantic, "_fit_hdbscan", side_effect=AssertionError()):
            result = semantic.search_semantic(self.vectors[:1], [1, 0], options={"use_umap": False, "use_hdbscan": False})
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["projection"])
        self.assertFalse(result["parameters"]["query_in_fit"])
        self.assertEqual(result["semantic_results"], result["baseline_results"])

    def test_cosine_projection_normalizes_for_formula_only_and_handles_zero(self):
        matrix = np.asarray([[2., 0.], [0., 5.], [0., 0.]])
        before = matrix.copy()
        np.testing.assert_allclose(semantic._projection_distances(matrix, np.asarray([10., 0.]), "cosine"), [0, 1, 1])
        np.testing.assert_allclose(semantic._projection_distances(matrix, np.asarray([0., 0.]), "cosine"), [1, 1, 1])
        np.testing.assert_array_equal(matrix, before)

    def test_options_reject_dendrogram_without_cluster_and_nonboolean_flags(self):
        for options in ({"use_hdbscan": False, "dendrogram": True}, {"use_umap": "false"}, {"use_hdbscan": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                semantic.search_semantic(self.vectors, [1, 0], options=options)

    @unittest.skipUnless(importlib.util.find_spec("umap") and importlib.util.find_spec("hdbscan"), "optional science dependencies")
    def test_real_standalone_stages_fit_independently(self):
        rng = np.random.default_rng(183)
        vectors = []
        for group in range(3):
            for i in range(12):
                center = np.eye(8)[group]
                vectors.append({"topic_id": f"g{group}_{i}", "category": "conteudo_fonte",
                                "embedding": (center + rng.normal(0, .02, 8)).tolist()})
        query = np.eye(8)[0]
        reduced = semantic.search_semantic(vectors, query, options={"use_hdbscan": False, "n_components": 3, "n_neighbors": 5})
        clustered = semantic.search_semantic(vectors, query, options={"use_umap": False, "dendrogram": True})
        self.assertEqual(reduced["status"], "ok")
        self.assertEqual(reduced["projection"]["dimensions"], 3)
        self.assertEqual(clustered["status"], "ok")
        self.assertTrue(all(r["topic_id"].startswith("g0") for r in clustered["semantic_results"]))
        self.assertEqual(len(clustered["dendrogram"]["single_linkage"]), len(vectors))


class IndependentComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ComparisonTests()
        self.fixture.setUp()
        self.project = self.fixture.project

    def tearDown(self):
        self.fixture.tearDown()

    def test_new_modes_save_export_reproduce_and_keep_exact_projection_coordinates(self):
        configs = [self.fixture.config("UMAP separado", mode="umap", n_components=2, metric="euclidean"),
                   self.fixture.config("HDBSCAN separado", mode="hdbscan", min_cluster_size=2),
                   self.fixture.config("Combinados", mode="semantic", n_components=2, min_cluster_size=2)]
        def combined(matrix, params):
            return projection(matrix, params), fake_model([0]*len(matrix))
        with patch.object(semantic, "_fit_umap", side_effect=projection), \
                patch.object(semantic, "_fit_hdbscan", side_effect=lambda matrix, params: fake_model([0]*len(matrix))), \
                patch.object(semantic, "_fit_clusters", side_effect=combined):
            preset = save_preset(self.project, {"name": "Etapas separadas", "experiments": configs})
            result = self.fixture.run_configs(preset["experiments"])
            reproduced = reproduce_comparison(self.project, result["comparison_id"])
        self.assertTrue(all(e["status"] == "ok" for e in result["experiments"]))
        self.assertEqual(result["experiments"], reproduced["experiments"])
        self.assertEqual([e["config"]["mode"] for e in result["experiments"]], ["umap", "hdbscan", "semantic"])
        self.assertTrue(all("use_umap" not in e["config"] for e in result["experiments"]))
        archive = zipfile.ZipFile(io.BytesIO(export_vector_bundle(self.project, result["comparison_id"], persist=False)))
        manifest = json.loads(archive.read("manifesto.json"))
        self.assertEqual(len(manifest["projections"]), 2)
        for file in manifest["files"]:
            self.assertEqual(hashlib.sha256(archive.read(file["path"])).hexdigest(), file["sha256"])
        for projection_record in manifest["projections"]:
            experiment = next(e for e in result["experiments"] if e["experiment_id"] == projection_record["experiment_id"])
            points = experiment["spaces"][0]["projection"]["points"]
            with np.load(io.BytesIO(archive.read(projection_record["file"])), allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["consulta"][0], points[-1]["coordinates"])
                np.testing.assert_array_equal(arrays["vetores"], [p["coordinates"] for p in points[:-1]])
                for association in manifest["associations"]:
                    if association.get("projection_file") != projection_record["file"]:
                        continue
                    vector = arrays["vetores"][association["projection_row"]]
                    query = arrays["consulta"][0]
                    self.assertAlmostEqual(association["projected_euclidean_distance"], np.linalg.norm(vector-query))
                    if association["distance_space"] == "umap_projection":
                        self.assertAlmostEqual(association["vector_distance"], np.linalg.norm(vector-query))
                    else:
                        self.assertAlmostEqual(association["vector_distance"], association["original_distance"])
        csv_rows = list(csv.DictReader(io.StringIO(archive.read("metricas.csv").decode("utf-8-sig"))))
        self.assertTrue(any(r["distance_space"] == "umap_projection" for r in csv_rows))
        self.assertIn("distance_space", export_comparison(self.project, result["comparison_id"], "csv").splitlines()[0])

    def test_umap_rejects_dendrogram_but_hdbscan_accepts_it(self):
        with self.assertRaisesRegex(ValueError, "dendrograma"):
            validate_config(self.project, self.fixture.config(mode="umap", dendrogram=True))
        result = validate_config(self.project, self.fixture.config(mode="hdbscan", dendrogram=True))
        self.assertTrue(result["dendrogram"])


if __name__ == "__main__":
    unittest.main()
