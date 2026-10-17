import os
import unittest
from unittest.mock import patch

import numpy as np

from autorag.providers import OllamaProvider


class OllamaCloudRuntimeTests(unittest.TestCase):
    def provider(self, gemma=False):
        provider = object.__new__(OllamaProvider)
        provider.profile = {"modelo": "embeddinggemma:300m-qat-q4_0" if gemma else "bge-m3",
                            "endpoint": "http://127.0.0.1:11434", "dimensoes": 2}
        if gemma:
            provider.profile["reducao"] = "matryoshka"
        return provider

    def test_cpu_batches_preserve_vectors_and_unload_only_after_final_batch(self):
        calls = []

        def respond(url, payload):
            calls.append(payload)
            return {"embeddings": [[float(value), float(value) + 1] for value in payload["input"]]}

        with patch.dict(os.environ, {"AUTORAG_OLLAMA_INPUT_BATCH": "1", "AUTORAG_OLLAMA_KEEP_ALIVE": "0"}), \
                patch("autorag.providers.http_json", side_effect=respond):
            actual = self.provider().embed(["1", "2", "3"])
        np.testing.assert_array_equal(actual, [[1, 2], [2, 3], [3, 4]])
        self.assertEqual([row["keep_alive"] for row in calls], ["5m", "5m", 0])
        self.assertTrue(all(row["truncate"] is False for row in calls))

    def test_desktop_defaults_keep_bulk_request_and_existing_lifetime(self):
        with patch.dict(os.environ, {}, clear=True), patch("autorag.providers.http_json", return_value={"embeddings": [[1, 2], [3, 4]]}) as request:
            actual = self.provider().embed(["one", "two"])
        np.testing.assert_array_equal(actual, [[1, 2], [3, 4]])
        self.assertEqual(request.call_args.args[1]["input"], ["one", "two"])
        self.assertEqual(request.call_args.args[1]["keep_alive"], "5m")

    def test_gemma_reduction_and_query_preparation_are_unchanged(self):
        with patch.dict(os.environ, {"AUTORAG_OLLAMA_INPUT_BATCH": "1", "AUTORAG_OLLAMA_KEEP_ALIVE": "0"}), \
                patch("autorag.providers.http_json", return_value={"embeddings": [[1, 2, 3]]}) as request:
            actual = self.provider(gemma=True).embed(["question"], role="consulta")
        np.testing.assert_array_equal(actual, [[1, 2]])
        self.assertEqual(request.call_args.args[1]["input"], ["task: search result | query: question"])

    def test_bad_batch_configuration_fails_before_calling_provider(self):
        with patch.dict(os.environ, {"AUTORAG_OLLAMA_INPUT_BATCH": "-1"}), \
                patch("autorag.providers.http_json") as request:
            with self.assertRaises(ValueError):
                self.provider().embed(["question"])
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
