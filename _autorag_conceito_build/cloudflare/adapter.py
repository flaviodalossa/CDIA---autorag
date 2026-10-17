"""Keep accepted jobs and confirmed results durable on ephemeral container disks."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hmac
import json
import os
import threading

from fastapi import HTTPException
from fastapi.responses import JSONResponse


def install(api, store):
    mutation = threading.Lock()
    dirty = threading.Event()

    def write_job(record):
        # The R2 copy is acknowledged before a terminal state is published in RAM.
        store.save_job(record)
        api.write_json(api.project.root / "jobs" / (record["job_id"] + ".json"), record)

    def submit(kind, function):
        with api.job_lock:
            if sum(job["status"] in {"queued", "running"} for job in api.jobs.values()) >= api.MAX_JOBS:
                raise HTTPException(429, "Fila cheia. Aguarde a conclusão das operações.")
            job_id = api.uid("job")
            record = {"job_id": job_id, "kind": kind, "status": "queued", "created_at": api.now()}
            api.jobs[job_id] = record
            try:
                write_job(record)
            except Exception as error:
                api.jobs.pop(job_id, None)
                raise HTTPException(503, "Não foi possível confirmar a fila no R2. Tente novamente.") from error

        def work():
            terminal = None
            try:
                with api.job_lock:
                    record.update(status="running", started_at=api.now())
                    current = deepcopy(record)
                write_job(current)
                with mutation:
                    result = function()
                    store.checkpoint(api.project)
                    terminal = {**deepcopy(current), "result": result, "status": "succeeded", "finished_at": api.now()}
                    write_job(terminal)
            except Exception as error:
                terminal = {**deepcopy(record), "status": "failed", "error": str(error), "finished_at": api.now()}
                try:
                    with mutation:
                        store.checkpoint(api.project)
                    write_job(terminal)
                except Exception:
                    terminal["persistence_warning"] = "A operação não foi confirmada no R2. O serviço tentará persistir novamente antes de suspender."
                    dirty.set()
            finally:
                with api.job_lock:
                    record.clear()
                    record.update(terminal)
                    if "persistence_warning" not in terminal:
                        api.jobs.pop(job_id, None)
        try:
            api.executor.submit(work)
        except Exception as error:
            terminal = {**record, "status": "failed", "error": "A fila foi interrompida. Reenvie a operação.", "finished_at": api.now()}
            write_job(terminal)
            with api.job_lock:
                api.jobs.pop(job_id, None)
            raise HTTPException(503, terminal["error"]) from error
        return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)

    api.submit = submit

    def flush():
        with mutation:
            store.checkpoint(api.project)
            with api.job_lock:
                pending = [deepcopy(job) for job in api.jobs.values() if job["status"] in {"failed", "succeeded"}]
            for job in pending:
                job.pop("persistence_warning", None)
                write_job(job)
                with api.job_lock:
                    api.jobs.pop(job["job_id"], None)
            dirty.clear()

    def authorized(scope):
        supplied = dict(scope.get("headers", [])).get(b"authorization", b"").decode("latin-1")
        expected = os.environ.get("AUTORAG_API_TOKEN", "")
        return bool(expected) and hmac.compare_digest(supplied, "Bearer " + expected)

    class CloudApp:
        async def __call__(self, scope, receive, send):
            if scope["type"] == "lifespan":
                async def startup_send(message):
                    if message["type"] == "lifespan.startup.complete":
                        # The original lifespan marks restored, interrupted jobs failed.
                        for path in (api.project.root / "jobs").glob("*.json"):
                            record = api.read_json(path)
                            if record.get("status") == "failed":
                                await asyncio.to_thread(store.save_job, record)
                    await send(message)
                return await api.app(scope, receive, startup_send)
            if scope["type"] != "http":
                return await api.app(scope, receive, send)
            path = scope.get("path", "")
            if path.startswith("/internal/"):
                if not authorized(scope):
                    return await JSONResponse({"detail": "Não autorizado."}, status_code=401)(scope, receive, send)
                if path == "/internal/cloud-state" and scope["method"] == "GET":
                    with api.job_lock:
                        active = any(job["status"] in {"queued", "running"} for job in api.jobs.values())
                    return await JSONResponse({"active": active or mutation.locked(), "pending_storage": dirty.is_set(), "last_checkpoint": store.last_checkpoint})(scope, receive, send)
                if path == "/internal/checkpoint" and scope["method"] == "POST":
                    if mutation.locked():
                        return await JSONResponse({"detail": "Processamento ativo."}, status_code=409)(scope, receive, send)
                    try:
                        await asyncio.to_thread(flush)
                    except Exception:
                        dirty.set()
                        return await JSONResponse({"detail": "Persistência R2 indisponível."}, status_code=503)(scope, receive, send)
                    return await JSONResponse({"status": "persisted"})(scope, receive, send)
                return await JSONResponse({"detail": "Não encontrado."}, status_code=404)(scope, receive, send)
            if path == "/api/profiles" and scope["method"] == "POST" and authorized(scope):
                if not mutation.acquire(blocking=False):
                    return await JSONResponse({"detail": "Há processamento em andamento. Aguarde para cadastrar o perfil."}, status_code=409)(scope, receive, send)
                messages = []
                async def buffered_send(message):
                    messages.append(message)
                try:
                    await api.app(scope, receive, buffered_send)
                    status = next((m["status"] for m in messages if m["type"] == "http.response.start"), 500)
                    if 200 <= status < 300:
                        try:
                            await asyncio.to_thread(store.checkpoint, api.project)
                        except Exception:
                            dirty.set()
                            return await JSONResponse({"detail": "O perfil não foi confirmado no R2. Tente novamente."}, status_code=503)(scope, receive, send)
                    for message in messages:
                        await send(message)
                finally:
                    mutation.release()
                return
            return await api.app(scope, receive, send)

    return CloudApp()
