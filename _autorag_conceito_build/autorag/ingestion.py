"""Generic, bounded preprocessing of normative documents.

This module never writes files or calls a model. The default enrichment is an
explicitly labelled extractive preview: the summary and answer copy the source,
and the suggested question is a template. A reviewed enrichment callback can be
supplied to generate richer representations independently of document metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.client
import io
import ipaddress
import json
from pathlib import Path
import re
import queue
import socket
import threading
import ssl
import time
from typing import Any, Callable, Mapping
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET
import zipfile


class IngestionError(ValueError):
    """Invalid source, unavailable extraction dependency, or exceeded limit."""


@dataclass(frozen=True)
class IngestionOptions:
    # Local paths are disabled until the host explicitly authorizes roots.
    allowed_roots: tuple[str | Path, ...] = ()
    max_file_bytes: int = 20 * 1024 * 1024
    max_total_bytes: int = 60 * 1024 * 1024
    max_text_chars: int = 2_000_000
    max_sources: int = 30
    max_topics: int = 2000
    chunk_chars: int = 4000
    request_timeout: float = 15.0
    total_timeout: float = 45.0
    max_redirects: int = 3
    max_pdf_pages: int = 1000
    max_docx_uncompressed_bytes: int = 40 * 1024 * 1024
    allow_http: bool = False

    def __post_init__(self):
        integer_fields = (
            "max_file_bytes", "max_total_bytes", "max_text_chars", "max_sources",
            "max_topics", "chunk_chars", "max_pdf_pages", "max_docx_uncompressed_bytes",
        )
        if any(isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int)
               or getattr(self, key) <= 0 for key in integer_fields):
            raise IngestionError("Os limites de ingestão devem ser inteiros positivos.")
        if self.chunk_chars < 200:
            raise IngestionError("O tamanho de tópico deve ter pelo menos 200 caracteres.")
        if not 0 <= self.max_redirects <= 10:
            raise IngestionError("O limite de redirecionamentos deve ficar entre 0 e 10.")
        if not 0 < self.request_timeout <= 120 or not 0 < self.total_timeout <= 300:
            raise IngestionError("Timeout de ingestão inválido.")


@dataclass
class IngestionResult:
    documents: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    @property
    def topics(self):
        return [topic for document in self.documents for topic in document["estrutura"]]

    def to_dict(self):
        return {
            "documentos": self.documents, "topicos": self.topics,
            "avisos": self.warnings, "fontes": self.sources,
        }

    def to_base(self, base_id="base_normativa"):
        return {
            "schema_version": "3.0", "base_id": base_id,
            "data_atualizacao_base": _now(),
            "ordem_documentos": [doc["documento_id"] for doc in self.documents],
            "documentos": self.documents,
            "estatisticas": {"documentos": len(self.documents),
                             "topicos_ativos": len(self.topics)},
        }


Enricher = Callable[[str, dict[str, Any]], Mapping[str, Any]]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _metadata(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise IngestionError("Os metadados devem ser um objeto JSON.")
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, allow_nan=False)
        if len(encoded) > 64_000:
            raise IngestionError("Metadados excedem 64 mil caracteres.")
        result = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise IngestionError("Os metadados devem conter apenas valores JSON finitos.") from exc
    for group in ("identificacao", "metadados", "vigencia"):
        if group in result and not isinstance(result[group], dict):
            raise IngestionError(f"O grupo de metadados {group} deve ser um objeto.")
    return result


def _local_path(source, options):
    if not options.allowed_roots:
        raise IngestionError("Caminhos locais estão desativados. Configure as raízes autorizadas no servidor.")
    path = Path(source).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in options.allowed_roots]
    if not any(path.is_relative_to(root) for root in roots):
        raise IngestionError("O arquivo está fora das raízes locais autorizadas.")
    if not path.is_file():
        raise IngestionError("O caminho deve apontar para um arquivo existente.")
    if path.stat().st_size > options.max_file_bytes:
        raise IngestionError("O arquivo excede o limite de tamanho.")
    return path


def _resolve_addresses(hostname, port, timeout):
    # A daemon avoids waiting for an OS DNS resolver beyond the request budget.
    mailbox = queue.Queue(maxsize=1)
    def resolve():
        try:
            mailbox.put((socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM), None))
        except (OSError, UnicodeError) as exc:
            mailbox.put((None, exc))
    threading.Thread(target=resolve, daemon=True).start()
    try:
        answers, error = mailbox.get(timeout=timeout)
    except queue.Empty as exc:
        raise IngestionError("A resolução DNS excedeu o tempo máximo.") from exc
    if error:
        raise IngestionError("Não foi possível resolver o host da URL.") from error
    return list(dict.fromkeys(answer[4][0] for answer in answers))


def _validated_url(url, options):
    if not isinstance(url, str) or len(url) > 8192 or any(ord(char) < 33 for char in url):
        raise IngestionError("URL inválida.")
    try:
        parsed = urlsplit(url)
        hostname, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise IngestionError("URL inválida.") from exc
    if parsed.scheme not in ({"https", "http"} if options.allow_http else {"https"}):
        raise IngestionError("Use uma URL HTTPS pública.")
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise IngestionError("A URL deve ter um host público e não pode conter credenciais.")
    if parsed.fragment:
        parsed = parsed._replace(fragment="")
    expected_port = 443 if parsed.scheme == "https" else 80
    if port is not None and port != expected_port:
        raise IngestionError("A URL deve usar a porta padrão HTTP ou HTTPS.")
    hostname = hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise IngestionError("Hosts locais ou privados não são permitidos.")
    addresses = _resolve_addresses(hostname, port or expected_port,
                                   min(options.request_timeout, options.total_timeout))
    if not addresses:
        raise IngestionError("O host não possui endereço público.")
    for address in addresses:
        try:
            value = ipaddress.ip_address(address)
        except ValueError as exc:
            raise IngestionError("O host retornou um endereço inválido.") from exc
        if not value.is_global or value.is_multicast or value.is_unspecified:
            raise IngestionError("URLs para redes privadas, locais ou reservadas não são permitidas.")
    return parsed, addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Use the already-validated address while verifying TLS against the host.

    No proxy/environment resolution and no second DNS lookup are performed.
    """
    def __init__(self, hostname, address, timeout):
        super().__init__(hostname, timeout=timeout, context=ssl.create_default_context())
        self._validated_address = address

    def connect(self):
        raw = socket.create_connection((self._validated_address, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, hostname, address, timeout):
        super().__init__(hostname, timeout=timeout)
        self._validated_address = address

    def connect(self):
        self.sock = socket.create_connection((self._validated_address, self.port), self.timeout)


def _download(url, options):
    started = time.monotonic()
    current = url
    for redirect in range(options.max_redirects + 1):
        parsed, addresses = _validated_url(current, options)
        remaining = options.total_timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise IngestionError("A obtenção da URL excedeu o tempo máximo.")
        timeout = min(options.request_timeout, remaining)
        connection_type = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
        connection = connection_type(parsed.hostname.rstrip("."), addresses[0], timeout)
        try:
            path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
            if parsed.query:
                path += "?" + quote(parsed.query, safe="=&%/:?@!$'()*+,;+-._~")
            connection.request("GET", path, headers={
                "User-Agent": "AutoRAG-Normativo/1.0", "Accept-Encoding": "identity",
                "Accept": "application/pdf,text/plain,text/html,application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            })
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location or redirect == options.max_redirects:
                    raise IngestionError("Redirecionamento ausente ou limite excedido.")
                current = urljoin(current, location)
                # The next iteration revalidates DNS, scheme and destination.
                continue
            if not 200 <= response.status < 300:
                raise IngestionError(f"A URL retornou HTTP {response.status}.")
            encoding = response.getheader("Content-Encoding", "identity").lower()
            if encoding not in {"identity", ""}:
                raise IngestionError("A URL retornou uma codificação compactada não solicitada.")
            size_header = response.getheader("Content-Length")
            if size_header:
                try:
                    declared_size = int(size_header)
                except ValueError as exc:
                    raise IngestionError("A URL retornou tamanho inválido.") from exc
                if declared_size < 0 or declared_size > options.max_file_bytes:
                    raise IngestionError("O documento remoto excede o limite de tamanho.")
            payload = bytearray()
            while True:
                if time.monotonic() - started > options.total_timeout:
                    raise IngestionError("A obtenção da URL excedeu o tempo máximo.")
                if connection.sock is not None:
                    connection.sock.settimeout(min(options.request_timeout,
                        max(0.001, options.total_timeout - (time.monotonic() - started))))
                block = response.read(min(65536, options.max_file_bytes + 1 - len(payload)))
                if not block:
                    break
                payload.extend(block)
                if len(payload) > options.max_file_bytes:
                    raise IngestionError("O documento remoto excede o limite de tamanho.")
            name = Path(unquote(parsed.path)).name or "documento"
            return bytes(payload), name, response.getheader("Content-Type", ""), urlunsplit(parsed)
        except (OSError, http.client.HTTPException) as exc:
            raise IngestionError(f"Não foi possível obter o documento remoto: {type(exc).__name__}.") from exc
        finally:
            connection.close()
    raise IngestionError("Limite de redirecionamentos excedido.")


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title_parts = [], []
        self.hidden = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if not self.hidden and tag in {"p", "div", "br", "li", "article", "section",
                                      "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "template"} and self.hidden:
            self.hidden -= 1
        if tag == "title":
            self.in_title = False
        if not self.hidden and tag in {"p", "div", "li", "article", "section",
                                      "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        elif not self.hidden:
            self.parts.append(data)


def _decode(data):
    if b"\x00" in data[:8192] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise IngestionError("O documento parece binário; forneça PDF, DOCX, HTML, TXT ou Markdown.")
    for encoding in ("utf-8-sig", "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestionError("Não foi possível decodificar o texto do documento.")


def _clean(text, options):
    # Normalize layout, preserving the words and all substantive content.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > options.max_text_chars:
        raise IngestionError("O texto extraído excede o limite de caracteres.")
    if not text:
        raise IngestionError("Nenhum texto extraível. PDFs digitalizados precisam passar por OCR antes da importação.")
    return text


def _extract(data, filename, content_type, options):
    suffix = Path(filename).suffix.lower()
    mime = content_type.split(";", 1)[0].strip().lower()
    title = ""
    if data.startswith(b"%PDF-") or suffix == ".pdf" or mime == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise IngestionError("Instale requirements-extracao.txt para extrair PDF (pypdf).") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise IngestionError("PDF criptografado não é aceito.")
            if len(reader.pages) > options.max_pdf_pages:
                raise IngestionError("O PDF excede o limite de páginas.")
            parts, chars = [], 0
            for page in reader.pages:
                part = page.extract_text() or ""
                chars += len(part)
                if chars > options.max_text_chars:
                    raise IngestionError("O PDF excede o limite de texto extraído.")
                parts.append(part)
            text = "\n\n".join(parts)
            title = str((reader.metadata or {}).get("/Title") or "")
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError("Não foi possível extrair o PDF. Verifique o arquivo ou aplique OCR.") from exc
        kind = "pdf"
    elif suffix == ".docx" or mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as package:
                infos = package.infolist()
                if len(infos) > 5000 or sum(info.file_size for info in infos) > options.max_docx_uncompressed_bytes:
                    raise IngestionError("O DOCX excede o limite de descompactação.")
                xml = package.read("word/document.xml")
                if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
                    raise IngestionError("Entidades externas não são aceitas no DOCX.")
                root = ET.fromstring(xml)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = []
                for paragraph in root.findall(".//w:p", ns):
                    paragraphs.append("".join(node.text or "" for node in paragraph.findall(".//w:t", ns)))
                text = "\n\n".join(paragraphs)
        except IngestionError:
            raise
        except (zipfile.BadZipFile, KeyError, ET.ParseError, RuntimeError) as exc:
            raise IngestionError("O arquivo DOCX é inválido.") from exc
        kind = "docx"
    elif suffix in {".html", ".htm", ".xhtml"} or mime in {"text/html", "application/xhtml+xml"}:
        parser = _HTMLText()
        parser.feed(_decode(data))
        text = "".join(parser.parts)
        title = "".join(parser.title_parts).strip()
        kind = "html"
    elif suffix in {".txt", ".md", ".markdown", ".csv", ".json", ".xml", ""} or mime in {"text/plain", "text/markdown"}:
        text = _decode(data)
        kind = "markdown" if suffix in {".md", ".markdown"} else "text"
    else:
        raise IngestionError("Formato não suportado. Use PDF, DOCX, HTML, TXT ou Markdown.")
    return _clean(text, options), title.strip()[:300], kind


_HEADING = re.compile(
    r"(?im)^(?:#{1,6}\s+\S[^\n]*|(?:art(?:igo)?\.?\s*\d+[^\n]*|cap[ií]tulo\s+[IVXLCDM\d]+[^\n]*|"
    r"se[çc][aã]o\s+[IVXLCDM\d]+[^\n]*|t[ií]tulo\s+[IVXLCDM\d]+[^\n]*))"
)


def _chunks(text, budget):
    """Partition text without dropping characters; prefer normative headings."""
    headings = [match.start() for match in _HEADING.finditer(text)]
    start = 0
    while start < len(text):
        limit = min(len(text), start + budget)
        if limit < len(text):
            candidates = [point for point in headings if start + min(150, budget // 4) <= point <= limit]
            if candidates:
                limit = candidates[-1]
            else:
                for boundary in ("\n\n", "\n", " "):
                    point = text.rfind(boundary, start + budget // 2, limit)
                    if point > start:
                        limit = point + len(boundary)
                        break
        yield text[start:limit], start, limit
        start = limit


def _excerpt(text, budget=800):
    if len(text) <= budget:
        return text
    cut = max(text.rfind(". ", 0, budget), text.rfind("\n", 0, budget))
    if cut < budget // 2:
        cut = text.rfind(" ", 0, budget)
    return text[:cut if cut > 0 else budget].strip()


def _string_list(value, name):
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise IngestionError(f"O campo {name} deve ser uma lista de textos.")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def _enrichment(chunk, title, metadata, enrich):
    if enrich is None:
        excerpt = _excerpt(chunk)
        return {
            "resumo": excerpt,
            "perguntas_sugeridas": [f"O que o trecho «{title[:120]}» informa?"],
            "resposta_sintese": excerpt,
            "palavras_chave": [], "base_normativa": [],
        }, "extrativo_com_pergunta_modelo"
    output = _metadata(enrich(chunk, metadata))
    for key in ("resumo", "resposta_sintese"):
        if not isinstance(output.get(key), str) or not output[key].strip():
            raise IngestionError(f"O enriquecedor deve retornar {key} como texto não vazio.")
        if len(output[key]) > 12000:
            raise IngestionError("O texto do enriquecedor excede o limite.")
    for key in ("perguntas_sugeridas", "palavras_chave", "base_normativa"):
        output[key] = _string_list(output.get(key), key)
    if not output["perguntas_sugeridas"]:
        raise IngestionError("O enriquecedor deve retornar pelo menos uma pergunta.")
    return {key: output[key] for key in ("resumo", "resposta_sintese", "perguntas_sugeridas",
                                        "palavras_chave", "base_normativa")}, "enriquecedor_externo"


def _document(data, name, source, metadata, options, enrich, content_type=""):
    if not data:
        raise IngestionError("O documento está vazio.")
    if len(data) > options.max_file_bytes:
        raise IngestionError("O arquivo excede o limite de tamanho.")
    text, extracted_title, kind = _extract(data, name, content_type, options)
    explicit_id = metadata.get("documento_id")
    if explicit_id is not None and (not isinstance(explicit_id, str) or not re.fullmatch(r"[\w.-]{1,120}", explicit_id)):
        raise IngestionError("documento_id deve conter até 120 letras, números, pontos, hífens ou sublinhados.")
    doc_id = explicit_id or "doc_" + _hash(source)[:20]
    identity = dict(metadata.get("identificacao", {}))
    document_meta = dict(metadata.get("metadados", {}))
    validity = dict(metadata.get("vigencia", {}))
    for key in ("tipo_documento", "identificacao_documento", "data_documento", "descricao_documento"):
        if key in metadata:
            identity[key] = metadata[key]
    for key in ("idioma", "orgao", "jurisdicao", "eixos", "eixo", "tags"):
        if key in metadata:
            document_meta[key] = metadata[key]
    for key in ("inicio_vigencia", "fim_vigencia"):
        if key in metadata:
            validity[key] = metadata[key]
    title = metadata.get("nome_documento") or extracted_title or Path(name).stem or "Documento"
    if not isinstance(title, str):
        raise IngestionError("nome_documento deve ser texto.")
    known = {"documento_id", "nome_documento", "identificacao", "metadados", "vigencia",
             "tipo_documento", "identificacao_documento", "data_documento", "descricao_documento",
             "idioma", "orgao", "jurisdicao", "eixos", "eixo", "tags", "inicio_vigencia", "fim_vigencia"}
    document_meta.update({key: value for key, value in metadata.items() if key not in known})
    topics = []
    for number, (chunk, start, end) in enumerate(_chunks(text, options.chunk_chars), 1):
        if number > options.max_topics:
            raise IngestionError("O documento excede o limite de tópicos.")
        first_line = chunk.strip().splitlines()[0]
        topic_title = re.sub(r"^#{1,6}\s+", "", first_line).strip()
        topic_title = topic_title[:160] or f"{title} — trecho {number}"
        enrichment, mode = _enrichment(chunk, topic_title, metadata, enrich)
        topics.append({
            "topico_id": f"{doc_id}::t{number:04d}_{_hash(chunk)[:10]}",
            "titulo": topic_title, "ativo": True, "ordem_no_documento": number,
            "caminho_hierarquico": [title], "secao": topic_title,
            "conteudo_fonte": chunk, **enrichment,
            "pagina_inicio": None, "pagina_fim": None,
            "localizacao": {"tipo": "caracteres_texto_extraido", "inicio": start, "fim": end},
            "processamento": {"modo": mode, "revisao_humana": "pendente",
                              "resumo_extrativo": enrich is None,
                              "pergunta_em_template": enrich is None,
                              **({"llm": dict(enrich.provenance)} if getattr(enrich, "provenance", None) else {})},
        })
    document = {
        "documento_id": doc_id, "nome_documento": title,
        "identificacao": identity, "metadados": document_meta, "vigencia": validity,
        "estrutura": topics, "ativo": True,
        "fonte": {"referencia": source, "nome_arquivo": name, "formato": kind,
                  "sha256": _hash(data), "tamanho_bytes": len(data)},
        "processamento": {"extrator": "autorag.ingestion.v1", "data": _now(),
                          "modo": "extrativo_com_pergunta_modelo" if enrich is None else "enriquecedor_externo",
                          "revisao_humana": "pendente", "metadados_fornecidos": bool(metadata),
                          **({"llm": dict(enrich.provenance)} if getattr(enrich, "provenance", None) else {})},
    }
    warning = (
        "Pré-processamento extrativo: resumos e respostas são cópias do texto; perguntas usam um modelo de frase. "
        "Revise os tópicos ou use um enriquecedor antes de publicar a base."
        if enrich is None else "Conteúdo gerado pelo enriquecedor externo: revise a fidelidade ao documento."
    )
    return IngestionResult([document], [warning], [{
        "fonte": source, "nome": name, "formato": kind, "sha256": _hash(data),
        "tamanho_bytes": len(data), "caracteres_extraidos": len(text), "topicos": len(topics),
    }])


def process_upload(filename: str, data: bytes, metadata=None, *, options=None, enrich: Enricher | None = None):
    """Preview one uploaded document without persisting or trusting its path."""
    options = options or IngestionOptions()
    if not isinstance(filename, str) or not filename.strip() or len(filename) > 500:
        raise IngestionError("Forneça um nome de arquivo válido.")
    if not isinstance(data, (bytes, bytearray)):
        raise IngestionError("O upload deve ser fornecido em bytes.")
    name = filename.replace("\\", "/").split("/")[-1]
    if name in {"", ".", ".."}:
        raise IngestionError("Nome de arquivo inválido.")
    return _document(bytes(data), name, "upload:" + name, _metadata(metadata), options, enrich)


def process_document(source: str, metadata=None, *, options=None, enrich: Enricher | None = None):
    """Preview a public URL or an authorized local path."""
    return ingest_sources([source], metadata, options=options, enrich=enrich)


def ingest_sources(sources: list[str], metadata=None, *, options=None, enrich: Enricher | None = None):
    """Preview sources with optional metadata supplied independently.

    metadata applies to all documents. For per-document values call
    process_document/process_upload separately and combine their documents.
    Paths are disabled by default; URLs permit only public HTTPS destinations.
    Failures raise IngestionError and no partial result is published.
    """
    options = options or IngestionOptions()
    meta = _metadata(metadata)
    if not isinstance(sources, (list, tuple)) or not sources or len(sources) > options.max_sources:
        raise IngestionError(f"Forneça de 1 a {options.max_sources} caminhos ou URLs.")
    result, total_bytes = IngestionResult(), 0
    seen = set()
    for source in sources:
        if not isinstance(source, str) or not source.strip():
            raise IngestionError("Cada fonte deve ser um caminho ou URL não vazio.")
        source = source.strip()
        if re.match(r"^[a-z][a-z0-9+.-]*://", source, re.I):
            data, name, mime, canonical = _download(source, options)
        else:
            path = _local_path(source, options)
            with path.open("rb") as stream:
                data = stream.read(options.max_file_bytes + 1)
            name, mime, canonical = path.name, "", str(path)
        total_bytes += len(data)
        if total_bytes > options.max_total_bytes:
            raise IngestionError("O lote excede o limite total de bytes.")
        item = _document(data, name, canonical, meta, options, enrich, mime)
        doc_id = item.documents[0]["documento_id"]
        if doc_id in seen:
            raise IngestionError("Fonte ou documento_id duplicado no lote.")
        seen.add(doc_id)
        result.documents.extend(item.documents)
        result.sources.extend(item.sources)
        result.warnings.extend(warning for warning in item.warnings if warning not in result.warnings)
        if len(result.topics) > options.max_topics:
            raise IngestionError("O lote excede o limite total de tópicos.")
    for number, document in enumerate(result.documents, 1):
        document["ordem_documento"] = number
    return result
