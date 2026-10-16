from dataclasses import replace
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from autorag.core import category_text, topics
from autorag.ingestion import (
    IngestionError, IngestionOptions, _chunks, _download, _PinnedHTTPSConnection,
    _validated_url, ingest_sources, process_document, process_upload,
)


def public_dns(*args, **kwargs):
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


class FakeResponse:
    def __init__(self, data=b"", status=200, headers=None):
        self.status, self.data, self.headers = status, io.BytesIO(data), headers or {}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, size):
        return self.data.read(size)


class FakeConnection:
    sock = None
    responses = []
    created = []
    requests = []

    def __init__(self, hostname, address, timeout):
        self.created.append((hostname, address, timeout))

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        return self.responses.pop(0)

    def close(self):
        pass


class IngestionTests(unittest.TestCase):
    def test_preview_without_metadata_is_compatible_and_explicit(self):
        result = process_upload("portaria.md", "# Contratações públicas\n\nArt. 1º Esta portaria define o procedimento.".encode())
        base = result.to_base()
        topic, doc = list(topics(base))[0]
        self.assertEqual(doc["nome_documento"], "portaria")
        self.assertEqual(doc["metadados"], {})
        self.assertEqual(topic["resumo"], topic["resposta_sintese"])
        self.assertTrue(topic["processamento"]["resumo_extrativo"])
        self.assertTrue(topic["processamento"]["pergunta_em_template"])
        self.assertEqual(topic["processamento"]["revisao_humana"], "pendente")
        self.assertTrue(category_text(topic, doc, "conteudo_fonte"))
        self.assertTrue(category_text(topic, doc, "perguntas_sugeridas"))
        self.assertTrue(category_text(topic, doc, "resposta_sintese"))
        self.assertEqual(len(result.to_dict()["topicos"]), 1)
        json.dumps(result.to_dict(), allow_nan=False)
        self.assertTrue(result.warnings)

    def test_independent_flat_and_nested_metadata(self):
        result = process_upload("norma.txt", b"Art. 1 A regra deve ser observada.", {
            "nome_documento": "Norma de licitação", "orgao": "Tribunal",
            "identificacao": {"tipo_documento": "Portaria"},
            "data_documento": "2025-01-03", "vigencia": {"fim_vigencia": None},
            "inicio_vigencia": "2025-01-04", "campo_personalizado": ["A", "B"],
        })
        doc = result.documents[0]
        self.assertEqual(doc["nome_documento"], "Norma de licitação")
        self.assertEqual(doc["identificacao"]["tipo_documento"], "Portaria")
        self.assertEqual(doc["identificacao"]["data_documento"], "2025-01-03")
        self.assertEqual(doc["vigencia"]["inicio_vigencia"], "2025-01-04")
        self.assertEqual(doc["metadados"]["campo_personalizado"], ["A", "B"])
        self.assertEqual(doc["metadados"]["orgao"], "Tribunal")

    def test_partition_is_complete_and_bounded(self):
        text = "Introdução.\n\n" + "\n\n".join(f"Art. {i} " + ("Texto normativo de licitação. " * 20) for i in range(20))
        chunks = list(_chunks(text, 500))
        self.assertEqual("".join(part for part, _, _ in chunks), text)
        self.assertTrue(all(end-start == len(part) <= 500 for part, start, end in chunks))
        self.assertEqual(chunks[-1][2], len(text))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous[2], current[1])

    def test_default_local_paths_disabled_then_explicit_root_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"norma.txt"
            path.write_text("Regra administrativa independente de ética.", encoding="utf-8")
            with self.assertRaisesRegex(IngestionError, "desativados"):
                process_document(str(path))
            options = IngestionOptions(allowed_roots=(folder,))
            result = process_document(str(path), options=options)
            self.assertEqual(result.documents[0]["estrutura"][0]["conteudo_fonte"], path.read_text(encoding="utf-8"))
            self.assertFalse((Path(folder)/"dados").exists())

    def test_local_path_cannot_escape_authorized_root(self):
        with tempfile.TemporaryDirectory() as folder:
            authorized = Path(folder)/"allowed"
            authorized.mkdir()
            outside = Path(folder)/"outside.txt"
            outside.write_text("fora", encoding="utf-8")
            with self.assertRaisesRegex(IngestionError, "fora das raízes"):
                process_document(str(authorized/".."/outside.name),
                                 options=IngestionOptions(allowed_roots=(authorized,)))

    def test_metadata_changes_preserve_document_and_topic_ids(self):
        a = process_upload("regulamento.txt", b"Art. 1 O servidor deve manter registro.")
        b = process_upload("regulamento.txt", b"Art. 1 O servidor deve manter registro.", {"orgao": "Outro"})
        self.assertEqual(a.documents[0]["documento_id"], b.documents[0]["documento_id"])
        self.assertEqual(a.topics[0]["topico_id"], b.topics[0]["topico_id"])

    def test_html_title_and_hidden_elements(self):
        html = b"<html><head><title>Manual de compras</title><style>secret style</style></head><body><h1>Regras</h1><p>Conteudo.</p><script>secret script</script></body></html>"
        result = process_upload("manual.html", html)
        self.assertEqual(result.documents[0]["nome_documento"], "Manual de compras")
        content = result.topics[0]["conteudo_fonte"]
        self.assertIn("Regras", content)
        self.assertNotIn("secret", content)
        self.assertNotIn("Manual de compras", content)

    @unittest.skipUnless(importlib.util.find_spec("pypdf"), "pypdf opcional indisponível")
    def test_pdf_text_extraction(self):
        stream = b"BT /F1 12 Tf 72 720 Td (Art. 1 Public procurement rule.) Tj ET"
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
                   b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                   b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
                   b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
        data = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for number, obj in enumerate(objects, 1):
            offsets.append(len(data))
            data.extend(str(number).encode() + b" 0 obj\n" + obj + b"\nendobj\n")
        xref = len(data)
        data.extend(b"xref\n0 6\n0000000000 65535 f \n")
        for offset in offsets[1:]:
            data.extend(f"{offset:010d} 00000 n \n".encode())
        data.extend(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n")
        result = process_upload("norma.pdf", bytes(data))
        self.assertIn("Public procurement rule.", result.topics[0]["conteudo_fonte"])
        self.assertEqual(result.sources[0]["formato"], "pdf")

    def test_docx_text_and_table_paragraphs(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as package:
            package.writestr("word/document.xml", """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Art. 1 Regras</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Conteúdo da tabela.</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>""")
        result = process_upload("manual.docx", stream.getvalue())
        self.assertIn("Conteúdo da tabela.", result.topics[0]["conteudo_fonte"])
        self.assertEqual(result.sources[0]["formato"], "docx")

    def test_docx_expansion_limit_and_xml_entity_rejected(self):
        for xml, options in [
            ("x" * 500, IngestionOptions(max_docx_uncompressed_bytes=10)),
            ('<!DOCTYPE foo [<!ENTITY x "forbidden">]><foo>&x;</foo>', IngestionOptions()),
        ]:
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as package:
                package.writestr("word/document.xml", xml)
            with self.assertRaises(IngestionError):
                process_upload("file.docx", stream.getvalue(), options=options)

    def test_limits_empty_binary_and_metadata_validation(self):
        cases = [
            lambda: process_upload("a.txt", b""),
            lambda: process_upload("a.txt", b"x" * 20, options=IngestionOptions(max_file_bytes=10)),
            lambda: process_upload("a.txt", b"x" * 20, options=IngestionOptions(max_text_chars=10)),
            lambda: process_upload("a.exe", b"MZfile"),
            lambda: process_upload("a.txt", b"abc\x00xyz"),
            lambda: process_upload("a.txt", b"text", {"metadados": []}),
            lambda: process_upload("a.txt", b"text", {"number": float("nan")}),
            lambda: process_upload("a.txt", b"text", {"documento_id": "../escape"}),
            lambda: ingest_sources([]),
        ]
        for action in cases:
            with self.subTest(action=action):
                with self.assertRaises(IngestionError):
                    action()

    def test_enrichment_is_injected_and_validated(self):
        seen = []
        def enrichment(text, metadata):
            seen.append((text, metadata))
            return {"resumo": "Resumo revisável.", "resposta_sintese": "Resposta revisável.",
                    "perguntas_sugeridas": ["Qual é a regra?"], "palavras_chave": ["licitação"]}
        result = process_upload("a.txt", b"Regra.", {"orgao": "X"}, enrich=enrichment)
        self.assertEqual(seen, [("Regra.", {"orgao": "X"})])
        self.assertEqual(result.topics[0]["processamento"]["modo"], "enriquecedor_externo")
        self.assertFalse(result.topics[0]["processamento"]["resumo_extrativo"])
        with self.assertRaises(IngestionError):
            process_upload("a.txt", b"Regra.", enrich=lambda *_: {"resumo": "incompleto"})

    def test_duplicate_sources_or_explicit_ids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder)/"1.txt", Path(folder)/"2.txt"
            first.write_text("Regra 1.", encoding="utf-8")
            second.write_text("Regra 2.", encoding="utf-8")
            options = IngestionOptions(allowed_roots=(folder,))
            for sources, metadata in [([str(first), str(first)], None),
                                      ([str(first), str(second)], {"documento_id": "same"})]:
                with self.assertRaisesRegex(IngestionError, "duplicado"):
                    ingest_sources(sources, metadata, options=options)

    def test_batch_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/f"{i}.txt" for i in range(2)]
            for path in paths:
                path.write_text("A" * 30, encoding="utf-8")
            options = IngestionOptions(allowed_roots=(folder,), max_total_bytes=50)
            with self.assertRaisesRegex(IngestionError, "total de bytes"):
                ingest_sources([str(path) for path in paths], options=options)
            with self.assertRaisesRegex(IngestionError, "total de tópicos"):
                ingest_sources([str(path) for path in paths], options=replace(options, max_total_bytes=100, max_topics=1))

    def test_private_or_unsafe_urls_blocked(self):
        for url in ["http://example.com/a.txt", "file:///etc/passwd",
                    "ftp://example.com/a.txt", "https://user:pass@example.com/a",
                    "https://example.com:8443/a", "https://localhost/a",
                    "https://host.local/a", "https://example.com/a\r\nHeader:x"]:
            with self.subTest(url=url):
                with self.assertRaises(IngestionError):
                    _validated_url(url, IngestionOptions())
        for address in ["127.0.0.1", "10.0.0.2", "169.254.169.254", "::1", "fd00::1", "224.0.0.1"]:
            with self.subTest(address=address), patch("autorag.ingestion.socket.getaddrinfo",
                    return_value=[(2, 1, 6, "", (address, 443))]):
                with self.assertRaises(IngestionError):
                    _validated_url("https://example.com/a", IngestionOptions())

    def test_mixed_public_private_dns_blocked(self):
        answers = public_dns() + [(2, 1, 6, "", ("192.168.1.5", 443))]
        with patch("autorag.ingestion.socket.getaddrinfo", return_value=answers):
            with self.assertRaises(IngestionError):
                _validated_url("https://example.com/a", IngestionOptions())

    @patch("autorag.ingestion.socket.getaddrinfo", side_effect=public_dns)
    @patch("autorag.ingestion._PinnedHTTPSConnection", FakeConnection)
    def test_public_download_is_bounded_and_pins_address(self, dns):
        FakeConnection.created = []
        FakeConnection.responses = [FakeResponse(b"Regra remota.", headers={"Content-Type": "text/plain"})]
        result = process_document("https://example.com/norma.txt")
        self.assertEqual(result.topics[0]["conteudo_fonte"], "Regra remota.")
        self.assertEqual(FakeConnection.created[0][:2], ("example.com", "93.184.216.34"))
        for response in [
            FakeResponse(b"A" * 20), FakeResponse(headers={"Content-Length": "20"}),
            FakeResponse(headers={"Content-Encoding": "gzip"}),
        ]:
            FakeConnection.responses = [response]
            with self.assertRaises(IngestionError):
                _download("https://example.com/a.txt", IngestionOptions(max_file_bytes=10))

    @patch("autorag.ingestion.socket.getaddrinfo", side_effect=public_dns)
    @patch("autorag.ingestion._PinnedHTTPSConnection", FakeConnection)
    def test_unicode_url_is_encoded_in_request(self, dns):
        FakeConnection.requests = []
        FakeConnection.responses = [FakeResponse(b"Regra.")]
        _download("https://example.com/licitação.txt?órgão=tribunal", IngestionOptions())
        self.assertEqual(FakeConnection.requests[0][0][1], "/licita%C3%A7%C3%A3o.txt?%C3%B3rg%C3%A3o=tribunal")

    @patch("autorag.ingestion._PinnedHTTPSConnection", FakeConnection)
    def test_redirect_revalidates_destination(self):
        def dns(host, *args, **kwargs):
            return [(2, 1, 6, "", ("127.0.0.1" if host == "evil.example" else "93.184.216.34", 443))]
        FakeConnection.responses = [FakeResponse(status=302, headers={"Location": "https://evil.example/metadata"})]
        with patch("autorag.ingestion.socket.getaddrinfo", side_effect=dns):
            with self.assertRaisesRegex(IngestionError, "privadas"):
                _download("https://example.com/a.txt", IngestionOptions())

    @patch("autorag.ingestion.socket.getaddrinfo", side_effect=public_dns)
    @patch("autorag.ingestion._PinnedHTTPSConnection", FakeConnection)
    def test_redirect_http_downgrade_and_redirect_loop_blocked(self, dns):
        FakeConnection.responses = [FakeResponse(status=302, headers={"Location": "http://example.com/a"})]
        with self.assertRaisesRegex(IngestionError, "HTTPS"):
            _download("https://example.com/a", IngestionOptions())
        FakeConnection.responses = [FakeResponse(status=302, headers={"Location": "/a"})] * 4
        with self.assertRaisesRegex(IngestionError, "limite excedido"):
            _download("https://example.com/a", IngestionOptions())

    def test_https_socket_uses_validated_ip_and_hostname_for_certificate(self):
        raw, tls = Mock(), Mock()
        context = Mock()
        context.wrap_socket.return_value = tls
        with patch("autorag.ingestion.ssl.create_default_context", return_value=context), \
             patch("autorag.ingestion.socket.create_connection", return_value=raw) as connect:
            connection = _PinnedHTTPSConnection("public.example", "93.184.216.34", 4)
            connection.connect()
            connect.assert_called_once_with(("93.184.216.34", 443), 4)
            context.wrap_socket.assert_called_once_with(raw, server_hostname="public.example")
            self.assertIs(connection.sock, tls)


if __name__ == "__main__":
    unittest.main()
