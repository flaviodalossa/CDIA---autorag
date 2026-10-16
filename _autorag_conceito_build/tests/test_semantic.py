"""Behavioral contracts for query-aware clustering and independent retrieval."""
import importlib.util
import json
import math
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from autorag.semantic import DEFAULT_CATEGORIES, _linkage_svg, search_semantic, vector_baseline


def row(topic, category, angle, suffix=""):
    return {"topic_id": topic, "category": category, "embedding_id": topic + "::" + category + suffix,
            "embedding": [math.cos(angle), math.sin(angle)], "text": topic + " original",
            "title": "Título " + topic, "source": {"url": "https://example.org/" + topic}}


def fake_model(labels):
    count = len(labels)
    # A valid scipy-format binary tree. It also records duplicate-point infinite
    # lambda values as produced by real HDBSCAN's condensed tree.
    linkage = np.asarray([[0, 1, .1, 2]] + [[count + i - 1, i + 1, .1 + i, i + 2]
                                        for i in range(1, count - 1)], dtype=float)
    condensed = np.asarray([(count, 0, float("inf"), 1), (count, 1, 10., 1)],
                           dtype=[("parent", int), ("child", int), ("lambda_val", float), ("child_size", int)])
    return SimpleNamespace(labels_=np.asarray(labels), probabilities_=np.ones(count),
                           single_linkage_tree_=SimpleNamespace(to_numpy=lambda: linkage),
                           condensed_tree_=SimpleNamespace(to_numpy=lambda: condensed))


class SemanticRankingTests(unittest.TestCase):
    def test_coverage_selects_then_original_distance_orders(self):
        vectors = [row("a", category, .7) for category in DEFAULT_CATEGORIES]
        vectors += [row("b", category, .3) for category in DEFAULT_CATEGORIES[:3]]
        vectors += [row("c", "resumo", .001, str(i)) for i in range(10)]
        # Four distinct categories beat three, which beat ten segments from one.
        # c is the closest vector but must not displace a or b at top_k=2.
        with patch("autorag.semantic._fit_clusters", return_value=(np.zeros((len(vectors)+1, 5)), fake_model([0]*(len(vectors)+1)))) as fit:
            result = search_semantic(vectors, [1, 0], top_k=2)
        self.assertEqual([r["topic_id"] for r in result["semantic_results"]], ["b", "a"])
        self.assertEqual([r["category_count"] for r in result["semantic_results"]], [3, 4])
        self.assertAlmostEqual(result["semantic_results"][0]["distance"], 1-math.cos(.3))
        # The query is the last row of the very same fit as the document vectors.
        np.testing.assert_allclose(fit.call_args.args[0][-1], [1, 0])
        self.assertEqual(fit.call_args.args[0].shape, (len(vectors)+1, 2))
        self.assertTrue(result["parameters"]["query_in_fit"])
        self.assertEqual(result["status"], "ok")
        json.dumps(result, allow_nan=False)

    def test_only_members_of_query_cluster_count(self):
        vectors = [row("a", category, .5) for category in DEFAULT_CATEGORIES]
        vectors += [row("b", "resumo", .2), row("b", "conteudo_fonte", .2)]
        # Only a's first vector shares the query cluster; its other categories
        # must not increase coverage, even though they belong to the same topic.
        labels = [0, 1, 1, 1, 0, 0, 0]
        with patch("autorag.semantic._fit_clusters", return_value=(np.zeros((7, 5)), fake_model(labels))):
            result = search_semantic(vectors, [1, 0], top_k=1)
        self.assertEqual(result["semantic_results"][0]["topic_id"], "b")
        self.assertEqual(result["semantic_results"][0]["categories"], ["conteudo_fonte", "resumo"])

    def test_noise_query_is_not_grouped_with_other_noise(self):
        vectors = [row(str(i), "conteudo_fonte", i*.1) for i in range(6)]
        with patch("autorag.semantic._fit_clusters", return_value=(np.zeros((7, 5)), fake_model([-1, -1, 0, 0, 0, 0, -1]))):
            result = search_semantic(vectors, [1, 0])
        self.assertEqual(result["status"], "query_noise")
        self.assertEqual(result["semantic_results"], [])
        self.assertEqual(len(result["baseline_results"]), 6)
        self.assertEqual(result["cluster"]["query_label"], -1)

    def test_arbitrary_category_combination_does_not_change_baseline_scope(self):
        vectors = [row(str(i), "custom", .3) for i in range(5)]
        vectors += [row("source", "conteudo_fonte", .4), row("nearest_summary", "resumo", 0)]
        with patch("autorag.semantic._fit_clusters", return_value=(np.zeros((6, 4)), fake_model([0]*6))) as fit:
            result = search_semantic(vectors, [1, 0], ["custom"])
        self.assertEqual(fit.call_args.args[0].shape, (6, 2))
        self.assertEqual(result["baseline_results"][0]["topic_id"], "source")
        self.assertEqual(len(result["semantic_results"]), 5)

    def test_small_or_empty_corpus_is_explicit_and_never_fakes_cluster(self):
        vectors = [row("a", "conteudo_fonte", 0), row("b", "resumo", .1)]
        with patch("autorag.semantic._fit_clusters") as fit:
            small = search_semantic(vectors, [1, 0])
            empty = search_semantic(vectors, [1, 0], ["missing"])
        fit.assert_not_called()
        self.assertEqual(small["status"], "insufficient_data")
        self.assertEqual(empty["status"], "empty")
        self.assertEqual(small["semantic_results"], [])
        self.assertEqual(small["baseline_results"][0]["topic_id"], "a")
        self.assertIsNone(small["cluster"]["query_label"])

    def test_baseline_pure_original_text_aggregates_segments_and_allows_all(self):
        vectors = [row("a", "conteudo_fonte", .5), row("a", "conteudo_fonte", .1, "second"),
                   row("b", "conteudo_fonte", .3), row("c", "resumo", 0)]
        with patch("autorag.semantic._fit_clusters") as fit:
            results = vector_baseline(vectors, [5, 0], top_k=None)
        fit.assert_not_called()
        self.assertEqual([r["topic_id"] for r in results], ["a", "b"])
        self.assertEqual(results[0]["vector_count"], 2)
        self.assertEqual(results[0]["category_count"], 1)
        self.assertEqual(results[0]["source"]["url"], "https://example.org/a")

    def test_supported_metrics_are_original_normalized_coordinates(self):
        vectors = [row("a", "conteudo_fonte", math.pi/2)]
        for metric, expected in [("cosine", 1), ("euclidean", math.sqrt(2)), ("manhattan", 2)]:
            self.assertAlmostEqual(vector_baseline(vectors, [1, 0], metric=metric)[0]["distance"], expected)

    def test_validation_rejects_invalid_embeddings_and_options(self):
        good = [row("a", "conteudo_fonte", 0)]
        for query in ([0, 0], [float("nan"), 0], [float("inf"), 0], [[1, 2]], [True, False], [1j, 1], []):
            with self.subTest(query=query), self.assertRaises(ValueError):
                search_semantic(good, query)
        for value in ([1], [0, 0], [1, float("nan")], ["1", "2"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                search_semantic([{**good[0], "embedding": value}], [1, 0])
        for source in ({"score": float("nan")}, {"array": np.asarray([1, 2])}, "invalid"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                search_semantic([{**good[0], "source": source}], [1, 0])
        for options in ({"metric": "invalid"}, {"n_neighbors": True}, {"min_cluster_size": 1},
                        {"min_dist": float("nan")}, {"random_state": -1}, {"unknown": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                search_semantic(good, [1, 0], options=options)
        for categories in ([], "resumo", [""]):
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                search_semantic(good, [1, 0], categories=categories)
        with self.assertRaises(ValueError):
            search_semantic(good*2, [1, 0])

    def test_dendrogram_leaf_positions_follow_tree_dfs_order(self):
        linkage = np.asarray([[0, 2, .1, 2], [1, 3, .2, 2], [4, 5, .3, 4]])
        svg = _linkage_svg(linkage, ["a", "b", "c", "query"])
        matches = re.findall(r'<text x="([0-9.]+)" y="[0-9.]+" text-anchor="middle"[^>]*><title>[^<]*</title>([0-9]+)</text>', svg)
        positions = {int(index): float(x) for x, index in matches}
        self.assertEqual(sorted(positions, key=positions.get), [0, 2, 1, 3])

    def test_dendrogram_exports_fitted_tree_and_json_safe_infinite_lambdas(self):
        vectors = [row(str(i), "resumo", .1) for i in range(5)]
        model = fake_model([0]*6)
        with patch("autorag.semantic._fit_clusters", return_value=(np.zeros((6, 4)), model)):
            result = search_semantic(vectors, [1, 0], options={"dendrogram": True})
        tree = result["dendrogram"]
        self.assertEqual(tree["single_linkage"], model.single_linkage_tree_.to_numpy().tolist())
        self.assertEqual(tree["point_labels"][-1], "__query__")
        self.assertTrue(tree["condensed_tree"][0]["lambda_infinite"])
        self.assertIsNone(tree["condensed_tree"][0]["lambda"])
        self.assertIn("<svg", tree["svg"])
        json.dumps(result, allow_nan=False)


@unittest.skipUnless(importlib.util.find_spec("umap") and importlib.util.find_spec("hdbscan"), "scientific dependencies optional in legacy environment")
class ScientificIntegrationTests(unittest.TestCase):
    def test_real_umap_hdbscan_is_repeatable_and_has_actual_hierarchy(self):
        rng = np.random.default_rng(123)
        vectors = []
        for group in range(3):
            center = np.zeros(8)
            center[group] = 1
            for topic in range(6):
                for category in DEFAULT_CATEGORIES:
                    vectors.append({"topic_id": f"g{group}t{topic}", "category": category,
                                    "embedding": (center + rng.normal(0, .025, 8)).tolist()})
        options = {"n_neighbors": 8, "min_cluster_size": 5, "dendrogram": True}
        first = search_semantic(vectors, [1, 0, 0, 0, 0, 0, 0, 0], options=options)
        second = search_semantic(vectors, [1, 0, 0, 0, 0, 0, 0, 0], options=options)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(len(first["semantic_results"]), 6)
        self.assertTrue(all(r["topic_id"].startswith("g0") for r in first["semantic_results"]))
        self.assertEqual(len(first["dendrogram"]["single_linkage"]), len(vectors))
        self.assertEqual(len(first["cluster"]["points"]), len(vectors)+1)
        json.dumps(first, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
