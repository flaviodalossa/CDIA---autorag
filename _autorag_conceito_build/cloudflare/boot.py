"""PID 1 supervisor for a private Ollama daemon and the existing ASGI application."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from .download_models import verify_installation

OLLAMA_URL = "http://127.0.0.1:11434"


def local_json(path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def start_ollama(stop_requested=lambda: False, timeout=90):
    root = Path(os.environ.get("OLLAMA_MODELS", "/opt/models/ollama"))
    lock = verify_installation(root)
    environment = os.environ.copy()
    # Ollama is available only to Python in this container, never as a public port.
    environment["OLLAMA_HOST"] = "127.0.0.1:11434"
    environment["OLLAMA_NO_CLOUD"] = "1"
    process = subprocess.Popen([os.environ.get("AUTORAG_OLLAMA_BIN", "/usr/bin/ollama"), "serve"], env=environment)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if stop_requested():
                raise InterruptedError("Startup interrupted.")
            if process.poll() is not None:
                raise RuntimeError("The private Ollama daemon exited during startup.")
            try:
                tags = local_json("/api/tags", timeout=2)
            except (OSError, ValueError):
                time.sleep(0.2)
                continue
            installed = {record["name"]: record["digest"] for record in tags.get("models", [])}
            for model in lock["models"]:
                name = model["name"] if ":" in model["name"] else model["name"] + ":latest"
                if installed.get(name) != model["manifest_sha256"]:
                    raise RuntimeError("The installed Ollama model differs from its audited digest: " + model["name"])
            return process
        raise TimeoutError("The private Ollama daemon did not become ready.")
    except BaseException:
        stop_processes([process])
        raise


def stop_processes(processes, timeout=30):
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    if len(os.environ.get("AUTORAG_API_TOKEN", "")) < 32:
        raise RuntimeError("Configure AUTORAG_API_TOKEN in the Worker before starting the backend.")
    stopping = False
    processes = []

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        ollama = start_ollama(lambda: stopping)
        processes.append(ollama)
        if stopping:
            return 0
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "cloudflare.runtime:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers"])
        processes.append(api)
        while not stopping:
            for process in processes:
                if process.poll() is not None:
                    return process.returncode if process.returncode != 0 else 1
            time.sleep(0.2)
        return 0
    except InterruptedError:
        return 0
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
