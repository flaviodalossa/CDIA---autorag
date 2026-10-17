"""AutoRAG HTTP entry point: python -m uvicorn api:app --host 127.0.0.1."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import re
import threading
import time

import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from autorag.core import CATEGORIES, Project, now, read_json, uid, write_json
from autorag.llm import LLMError
from autorag.providers import available_models, get_provider
from autorag.service import generate, register_profile, require_profile
from autorag.web_service import history, merge_documents, run_search, status

APP_ROOT = Path(__file__).resolve().parent
project = Project(os.getenv("AUTORAG_PROJECT_DIR", str(APP_ROOT)))
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autorag")
job_lock = threading.Lock()
jobs = {}
MAX_REQUEST_BYTES = 25 * 1024 * 1024
MAX_JOBS = 50


@asynccontextmanager
async def lifespan(app):
    if os.getenv("AUTORAG_ENV") == "production" and len(os.getenv("AUTORAG_API_TOKEN", "")) < 32:
        raise RuntimeError("Produção exige AUTORAG_API_TOKEN com pelo menos 32 caracteres.")
    # A single worker owns the local queue; interrupted jobs remain inspectable.
    for path in (project.root/"jobs").glob("*.json"):
        record = read_json(path)
        if record.get("status") in {"queued", "running"}:
            record.update(status="failed", error="Processamento interrompido pelo reinício do serviço. Reenvie a operação.", finished_at=now())
            write_json(path, record)
    yield


app = FastAPI(title="AutoRAG — bases normativas", version="0.1.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        expected = os.getenv("AUTORAG_API_TOKEN", "")
        supplied = request.headers.get("authorization", "")
        if expected and not hmac.compare_digest(supplied, "Bearer "+expected):
            return JSONResponse({"detail": "Informe o token de acesso nas configurações da aplicação."}, status_code=401)
        # Without a configured token, only a loopback client is accepted.
        if not expected and request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse({"detail": "Configure AUTORAG_API_TOKEN para acesso remoto."}, status_code=403)
        length = request.headers.get("content-length", "")
        if length and (not length.isdigit() or int(length) > MAX_REQUEST_BYTES):
            return JSONResponse({"detail": "Requisição excede o limite de 25 MiB."}, status_code=413)
        # Bound even chunked requests before multipart parsing allocates disk/memory.
        if request.method in {"POST", "PUT", "PATCH"}:
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    return JSONResponse({"detail": "Requisição excede o limite de 25 MiB."}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(LLMError)
async def invalid_llm(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


def submit(kind, function):
    with job_lock:
        if sum(j["status"] in {"queued", "running"} for j in jobs.values()) >= MAX_JOBS:
            raise HTTPException(429, "Fila cheia. Aguarde a conclusão das operações.")
        job_id = uid("job")
        record = {"job_id": job_id, "kind": kind, "status": "queued", "created_at": now()}
        jobs[job_id] = record
        try:
            write_json(project.root/"jobs"/(job_id+".json"), record)
        except OSError as exc:
            jobs.pop(job_id, None)
            raise HTTPException(503, "Não foi possível gravar a fila. Verifique o armazenamento do serviço.") from exc
    def work():
        try:
            with job_lock:
                record.update(status="running", started_at=now())
                snapshot = deepcopy(record)
            write_json(project.root/"jobs"/(job_id+".json"), snapshot)
            result = function()
            with job_lock:
                record.update(result=result, status="succeeded", finished_at=now())
        except Exception as exc:
            with job_lock:
                record.update(status="failed", error=str(exc), finished_at=now())
        finally:
            with job_lock:
                record["finished_at"] = now()
                snapshot = deepcopy(record)
            try:
                write_json(project.root/"jobs"/(job_id+".json"), snapshot)
            except Exception as exc:
                # Keep a terminal state available even if the disk is unavailable.
                with job_lock:
                    record["persistence_warning"] = str(exc)
            else:
                with job_lock:
                    jobs.pop(job_id, None)
    executor.submit(work)
    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/status")
def get_status():
    with project.lock():
        result = status(project)
    result["local_paths_enabled"] = bool(os.getenv("AUTORAG_ALLOWED_SOURCE_ROOTS"))
    return result


@app.get("/api/models")
def get_models():
    """Discover installations and credential presence without inference or secrets."""
    discovered = available_models(
        os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        str(project.root/"modelos"),
    )
    with project.lock():
        profiles = list(project.config()["perfis"].values())
    public = []
    fields = ("provedor", "modelo", "dimensoes", "dimensoes_nativas", "descricao", "custo", "reducao")
    for row in discovered:
        provider, model = row["provedor"], row["modelo"]
        available = bool(row.get("disponivel"))
        item = {key: row[key] for key in fields if key in row}
        item["disponivel"] = available
        if provider in {"openai", "gemini"}:
            item["detalhe"] = ("Credencial configurada; disponibilidade depende da conta e cota."
                               if available else "Configure a credencial desta API no servidor.")
        elif available:
            item["detalhe"] = "Modelo instalado no servidor de processamento."
        elif provider == "ollama":
            item["detalhe"] = "Modelo não instalado ou serviço Ollama indisponível."
        else:
            item["detalhe"] = ("Dependência de execução local não instalada."
                               if "dependência" in row.get("detalhe", "")
                               else "Modelo não instalado no servidor de processamento.")
        item["profile_ids"] = [p["perfil_id"] for p in profiles
                               if p["provedor"] == provider and
                               (p["modelo"].removesuffix(":latest") if provider == "ollama" else p["modelo"]) ==
                               (model.removesuffix(":latest") if provider == "ollama" else model)]
        public.append(item)
    return {"models": public, "checked_at": now()}


class ModelCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str = Field(min_length=1, max_length=100)


def check_local_model(profile):
    """Check both text preparation roles; never return vectors or touch the base."""
    if profile["provedor"] not in {"ollama", "sentence_transformers"}:
        raise ValueError("O teste automático aceita somente modelos locais; APIs externas não são chamadas.")
    started = time.perf_counter()
    provider = get_provider(profile)
    expected = profile["dimensoes"]
    norms = {}
    examples = (
        ("consulta", "Qual é o prazo de publicação de uma norma?"),
        ("documento", "Documento sintético de verificação: a norma deve ser publicada e mantida acessível."),
    )
    for role, value in examples:
        matrix = np.asarray(provider.embed([value], role=role), dtype=np.float64)
        if matrix.shape != (1, expected):
            raise ValueError(f"Dimensão inválida no teste de {role}: esperado um vetor de {expected} coordenadas.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"O teste de {role} retornou coordenadas não finitas.")
        norm = float(np.linalg.norm(matrix[0]))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"O teste de {role} retornou norma inválida.")
        norms[role] = norm
    return {"profile_id": profile["perfil_id"], "provider": profile["provedor"],
            "model": profile["modelo"], "quantity": 2, "dimensions": expected,
            "norms": norms, "finite": True,
            "latency_ms": round((time.perf_counter()-started)*1000, 2)}


@app.post("/api/models/check")
def check_model(body: ModelCheckRequest):
    with project.lock():
        profile = deepcopy(require_profile(project, body.profile_id))
    if profile["provedor"] not in {"ollama", "sentence_transformers"}:
        raise ValueError("O teste automático aceita somente modelos locais; APIs externas não são chamadas.")
    return submit("model_check", lambda: check_local_model(profile))


@app.get("/api/llms")
def get_llms():
    from autorag.llm import llm_catalog
    return llm_catalog()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if not re.fullmatch(r"job_[A-Za-z0-9_]+", job_id):
        raise HTTPException(404, "Operação não encontrada.")
    with job_lock:
        if job_id in jobs:
            return deepcopy(jobs[job_id])
    path = project.root/"jobs"/(job_id+".json")
    if not path.exists():
        raise HTTPException(404, "Operação não encontrada.")
    return read_json(path)



def _require_transfer_auth(request: Request):
    # Transfers expose complete document data: require an explicit configured
    # bearer even on loopback, in addition to the application API middleware.
    expected = os.getenv("AUTORAG_API_TOKEN", "")
    if not expected:
        raise HTTPException(403, "Importação e exportação exigem AUTORAG_API_TOKEN configurado.")
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied, "Bearer "+expected):
        raise HTTPException(401, "Token de acesso inválido.")


@app.post("/api/base/import")
async def import_prepared(request: Request):
    _require_transfer_auth(request)
    import asyncio
    from autorag.base_import import MAX_TRANSFER_BYTES, prepare_base_import, import_prepared_base
    async with request.form(max_files=2, max_fields=0) as form:
        entries = list(form.multi_items())
        if any(key not in {"base_file", "legacy_npz"} or not hasattr(value, "read") for key, value in entries):
            raise ValueError("Envie somente base_file JSON e legacy_npz opcional como arquivos.")
        if sum(key == "base_file" for key, _ in entries) != 1 or sum(key == "legacy_npz" for key, _ in entries) > 1:
            raise ValueError("Forneça exatamente um base_file e no máximo um legacy_npz.")
        payload = {}
        size = 0
        for key, value in entries:
            data = await value.read(MAX_TRANSFER_BYTES+1)
            size += len(data)
            if size > MAX_TRANSFER_BYTES:
                raise HTTPException(413, "A transferência excede 25 MiB.")
            payload[key] = data
    prepared = await asyncio.to_thread(prepare_base_import, payload["base_file"], payload.get("legacy_npz"))
    return submit("base_import", lambda: import_prepared_base(project, prepared))


@app.get("/api/base/export")
def export_prepared(request: Request):
    _require_transfer_auth(request)
    from autorag.base_import import canonical_digest, DIGEST_ALGORITHM
    with project.lock():
        if not project.base_path.exists():
            raise HTTPException(404, "Nenhuma base foi importada.")
        base = project.base()
        checksum = canonical_digest(base)
    return JSONResponse(base, headers={
        "Content-Disposition": 'attachment; filename="autorag-base.json"',
        "X-AutoRAG-Canonical-Digest": checksum,
        "X-AutoRAG-Digest-Algorithm": DIGEST_ALGORITHM,
    })


@app.post("/api/indices/import")
async def import_prepared_index(request: Request):
    _require_transfer_auth(request)
    import asyncio
    from autorag.base_import import MAX_TRANSFER_BYTES
    from autorag.index_transfer import validate_index_bundle, import_index_bundle
    async with request.form(max_files=1, max_fields=0) as form:
        entries = list(form.multi_items())
        if len(entries) != 1 or entries[0][0] != "index_file" or not hasattr(entries[0][1], "read"):
            raise ValueError("Forneça exatamente um arquivo index_file.")
        data = await entries[0][1].read(MAX_TRANSFER_BYTES+1)
        if len(data) > MAX_TRANSFER_BYTES:
            raise HTTPException(413, "A transferência excede 25 MiB.")
    await asyncio.to_thread(validate_index_bundle, project, data)
    return submit("index_import", lambda: import_index_bundle(project, data))

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=50000)
    profile_id: str = Field(min_length=1, max_length=100)
    categories: list[str] = Field(default_factory=lambda: ["perguntas_sugeridas", "resposta_sintese", "resumo", "conteudo_fonte"], min_length=1, max_length=8)
    mode: str = "semantic"
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    top_k: int = Field(default=6, ge=1, le=100)
    n_neighbors: int = Field(default=15, ge=2, le=200)
    n_components: int = Field(default=5, ge=2, le=50)
    min_dist: float = Field(default=0.0, ge=0, le=1)
    min_cluster_size: int = Field(default=5, ge=2, le=1000)
    min_samples: int = Field(default=2, ge=1, le=1000)
    random_state: int = Field(default=42, ge=0, le=2147483647)
    dendrogram: bool = False
    view: str = Field(default="topicos", pattern="^(topicos|vetores)$")
    minimum: float | None = Field(default=None, ge=-1, le=1)
    maximum: float | None = Field(default=None, ge=-1, le=1)
    partial: bool = False
    individual_limits: dict[str, float] = Field(default_factory=dict)
    individual_rule: str = Field(default="todos", pattern="^(todos|qualquer)$")


@app.post("/api/search")
def search(body: SearchRequest):
    if body.mode not in {"semantic", "laboratory"} or any(c not in CATEGORIES for c in body.categories):
        raise ValueError("Modo ou categoria desconhecida.")
    return submit("search", lambda: run_search(project, body.model_dump()))


@app.post("/api/ingest")
async def ingest(request: Request):
    from autorag.ingestion import IngestionOptions, ingest_sources, process_upload
    async with request.form(max_files=20, max_fields=20) as form:
        try:
            sources = json.loads(str(form.get("sources", "[]")))
            metadata = json.loads(str(form.get("metadata", "{}")))
        except (ValueError, TypeError) as exc:
            raise ValueError("Fontes e metadados devem ser JSON válido.") from exc
        if not isinstance(sources, list) or len(sources) > 30 or any(not isinstance(s, str) or len(s) > 4000 for s in sources):
            raise ValueError("Informe até 30 caminhos ou URLs.")
        if not isinstance(metadata, dict):
            raise ValueError("Metadados devem ser um objeto JSON.")
        metadata = dict(metadata)
        if metadata.get("title"):
            metadata.setdefault("nome_documento", metadata["title"])
            metadata.setdefault("descricao_documento", metadata["title"])
        commit = str(form.get("commit", "false")).lower() in {"true", "1"}
        preview_id = str(form.get("preview_id", ""))
        if preview_id:
            if not commit or not re.fullmatch(r"preview_[A-Za-z0-9_]+", preview_id):
                raise ValueError("Identificador de prévia inválido.")
            preview_path = project.root/"previews"/(preview_id+".json")
            if not preview_path.exists():
                raise ValueError("Prévia não encontrada. Prepare a base novamente.")
            reviewed = read_json(preview_path)
            return submit("ingest", lambda: {**merge_documents(project, reviewed["documents"]), "warnings": reviewed["warnings"], "preview_id": preview_id})
        from autorag.llm import make_enricher
        try:
            llm = json.loads(str(form.get("llm", "null")))
        except (ValueError, TypeError) as exc:
            raise ValueError("A configuração do LLM deve ser JSON válido.") from exc
        if llm is not None and not isinstance(llm, dict):
            raise ValueError("A configuração do LLM deve ser um objeto JSON.")
        uploads = []
        for key, value in form.multi_items():
            if hasattr(value, "read") and hasattr(value, "filename"):
                data = await value.read(20*1024*1024+1)
                if len(data) > 20*1024*1024:
                    raise ValueError("Cada arquivo pode ter até 20 MiB.")
                uploads.append((value.filename or "documento.txt", data))
        if not sources and not uploads:
            raise ValueError("Forneça pelo menos um documento por URL, caminho ou upload.")
        enrich = make_enricher(llm)
        allowed = tuple(Path(s.strip()).resolve() for s in os.getenv("AUTORAG_ALLOWED_SOURCE_ROOTS", "").split(os.pathsep) if s.strip())
        def work():
            options = IngestionOptions(allowed_roots=allowed)
            incoming, warnings = [], []
            if sources:
                result = ingest_sources(sources, metadata=metadata, options=options, enrich=enrich)
                incoming, warnings = list(result.documents), list(result.warnings)
            for filename, data in uploads:
                part = process_upload(filename, data, metadata=metadata, options=options, enrich=enrich)
                incoming.extend(part.documents)
                warnings.extend(part.warnings)
            ids = [d["documento_id"] for d in incoming]
            if len(ids) != len(set(ids)):
                raise ValueError("Fontes repetidas no mesmo lote. Use nomes distintos para os arquivos e remova URLs duplicadas.")
            if not incoming:
                raise ValueError("Nenhum documento pôde ser extraído. " + " ".join(warnings))
            payload = {"documents": len(incoming), "topics": sum(len(d.get("estrutura", [])) for d in incoming),
                       "warnings": warnings, "committed": False,
                       "preview": [{"document_id": d["documento_id"], "title": d.get("nome_documento"),
                                    "metadata": {"identificacao": d.get("identificacao"), "metadados": d.get("metadados"), "vigencia": d.get("vigencia")},
                                    "topics": d.get("estrutura", [])} for d in incoming]}
            if commit:
                payload.update(merge_documents(project, incoming))
            else:
                preview_id = uid("preview")
                with project.lock():
                    write_json(project.root/"previews"/(preview_id+".json"), {"documents": incoming, "warnings": warnings, "date": now()})
                payload["preview_id"] = preview_id
            return payload
        return submit("ingest", work)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str
    categories: list[str] = Field(min_length=1, max_length=8)
    operation: str = Field(default="atualizar", pattern="^(faltantes|atualizar|regenerar)$")
    simulate: bool = False
    batch_size: int = Field(default=16, ge=1, le=128)


@app.post("/api/generate")
def generate_vectors(body: GenerateRequest):
    return submit("generate", lambda: generate(project, body.profile_id, body.categories,
                   body.operation, batch_size=body.batch_size, simulate=body.simulate))


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    provider: str
    model: str = Field(min_length=1, max_length=200)
    dimensions: int = Field(ge=1, le=8192)
    endpoint: str | None = None


@app.post("/api/profiles")
def add_profile(body: ProfileRequest):
    endpoint = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if body.endpoint and body.endpoint.rstrip("/") != endpoint.rstrip("/"):
        raise ValueError("O endpoint Ollama deve corresponder a OLLAMA_BASE_URL configurado no servidor.")
    if body.provider not in {"openai", "gemini", "ollama", "sentence_transformers"}:
        raise ValueError("Provedor desconhecido.")
    return register_profile(project, body.provider, body.model, body.dimensions,
                            endpoint=endpoint if body.provider == "ollama" else None,
                            models_dir=str(project.root/"modelos"))


@app.get("/api/experiments")
def experiments():
    return {"experiments": history(project)}


@app.get("/api/experiments/{execution_id}")
def experiment(execution_id: str):
    if not re.fullmatch(r"(?:web|exp)_[A-Za-z0-9_]+", execution_id):
        raise HTTPException(404, "Experimento não encontrado.")
    folder = "web" if execution_id.startswith("web_") else "execucoes"
    path = project.root/"estudos"/folder/execution_id/"resultado.json"
    if not path.exists():
        raise HTTPException(404, "Experimento não encontrado.")
    return read_json(path)


app.mount("/", StaticFiles(directory=str(APP_ROOT/"web"), html=True), name="frontend")
