import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from autorag.ingestion import process_upload
from autorag.llm import LLMError, llm_catalog, make_enricher, _request


OUTPUT = {"resumo": "A norma exige análise prévia.", "resposta_sintese": "A análise deve ser prévia.",
          "perguntas_sugeridas": ["Quando analisar?"], "palavras_chave": ["análise"], "base_normativa": []}


class LLMTests(unittest.TestCase):
    def test_default_exact_luna_responses_schema_and_source_provenance(self):
        answer = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(OUTPUT)}]}]}
        with patch("autorag.llm.obter_chave", return_value="private-test-key"), \
                patch("autorag.llm._request", return_value=answer) as request:
            result = process_upload("norma.txt", b"Art. 1. A norma exige analise previa.",
                                    enrich=make_enricher())
        url, payload, key = request.call_args.args
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertEqual(result.topics[0]["conteudo_fonte"], "Art. 1. A norma exige analise previa.")
        self.assertEqual(result.topics[0]["processamento"]["llm"]["model"], "gpt-6-luna")
        self.assertNotIn(key, json.dumps(result.documents))
        self.assertNotIn("https://api", json.dumps(result.documents))

    def test_missing_key_fails_before_any_request_without_fallback(self):
        with patch("autorag.llm.obter_chave", return_value=""), patch("autorag.llm._request") as request:
            with self.assertRaisesRegex(LLMError, "OPENAI_API_KEY"):
                make_enricher()
            request.assert_not_called()
        self.assertIsNone(make_enricher({"provider": "extractive"}))

    def test_invalid_structured_output_refusal_and_incomplete_fail(self):
        responses = [
            {"status": "incomplete"},
            {"status": "completed", "output": [{"content": [{"type": "refusal"}]}]},
            {"status": "completed", "output": [{"content": [{"type": "output_text", "text": "not JSON"}]}]},
            {"status": "completed", "output": [{"content": [{"type": "output_text", "text": json.dumps({**OUTPUT, "conteudo_fonte": "inventado"})}]}]},
        ]
        for response in responses:
            with self.subTest(response=response), patch("autorag.llm.obter_chave", return_value="test"), \
                    patch("autorag.llm._request", return_value=response):
                with self.assertRaises(LLMError):
                    make_enricher()("Artigo fonte.", {})

    def test_ollama_json_chat_and_embedding_models_rejected(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://127.0.0.1:11434"}), \
                patch("autorag.llm._request", return_value={"done": True, "done_reason": "stop", "message": {"content": json.dumps(OUTPUT)}}) as request:
            enrich = make_enricher({"provider": "ollama", "model": "generative:test"})
            self.assertEqual(enrich("Trecho", {})["resumo"], OUTPUT["resumo"])
            self.assertTrue(request.call_args.args[0].endswith("/api/chat"))
            self.assertFalse(request.call_args.args[1]["stream"])
            self.assertEqual(request.call_args.args[1]["keep_alive"], 0)
            for model in ("bge-m3:latest", "embeddinggemma:300m-qat-q4_0"):
                with self.assertRaisesRegex(LLMError, "embeddings"):
                    make_enricher({"provider": "ollama", "model": model})

    def test_compatible_api_endpoint_comes_only_from_server(self):
        with patch.dict(os.environ, {"AUTORAG_LLM_BASE_URL": "https://example.test/v1", "AUTORAG_LLM_API_KEY": "secret"}), \
                patch("autorag.llm._request", return_value={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(OUTPUT)}}]}) as request:
            enrich = make_enricher({"provider": "openai_compatible", "model": "organization/open-model"})
            enrich("Texto", {})
            self.assertEqual(request.call_args.args[0], "https://example.test/v1/chat/completions")
            self.assertEqual(request.call_args.args[2], "secret")
        with self.assertRaises(LLMError):
            make_enricher({"provider": "ollama", "model": "x", "endpoint": "http://metadata"})

    def test_catalog_contains_no_credentials_or_endpoint(self):
        with patch("autorag.llm.obter_chave", return_value="private-test-key"), \
                patch.dict(os.environ, {"AUTORAG_LLM_BASE_URL": "https://private.test/v1", "AUTORAG_LLM_API_KEY": "private-test-key"}):
            catalog = llm_catalog()
        self.assertTrue(catalog["providers"][0]["available"])
        self.assertNotIn("private", json.dumps(catalog))

    def test_malformed_containers_and_unfinished_responses_are_rejected(self):
        for response in ({"status": "queued", "output": []},
                         {"status": "completed", "output": [None]},
                         {"status": "completed", "output": [{"content": [None]}]}):
            with patch("autorag.llm.obter_chave", return_value="test"), \
                    patch("autorag.llm._request", return_value=response):
                with self.assertRaises(LLMError):
                    make_enricher()("Texto", {})
        for response in ({"done": False}, {"done": True, "done_reason": "length"},
                         {"done": True, "done_reason": "stop", "message": None}):
            with patch("autorag.llm._request", return_value=response):
                with self.assertRaises(LLMError):
                    make_enricher({"provider": "ollama", "model": "generative:test"})("Texto", {})
        with self.assertRaises(LLMError):
            make_enricher({"provider": []})

    def test_invalid_credential_is_rejected_without_leaking_or_transmitting_it(self):
        with patch("urllib.request.OpenerDirector.open") as request:
            with self.assertRaises(LLMError) as caught:
                _request("https://example.test", {}, "secret\r\nheader")
            request.assert_not_called()
        self.assertNotIn("secret", str(caught.exception))

    def test_http_error_does_not_disclose_response_body(self):
        error = urllib.error.HTTPError("https://example.test", 401, "Unauthorized", {}, io.BytesIO(b"secret-body"))
        with patch("urllib.request.OpenerDirector.open", side_effect=error):
            with self.assertRaises(LLMError) as caught:
                _request("https://example.test", {}, "secret-key")
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
