"""Cloudflare container entry point; restore durable state before constructing the API."""
import os
from pathlib import Path

from .snapshot import R2State
from .adapter import install

if len(os.environ.get("AUTORAG_API_TOKEN", "")) < 32:
    raise RuntimeError("Configure AUTORAG_API_TOKEN com pelo menos 32 caracteres no Worker.")

store = R2State(Path(os.environ.get("AUTORAG_PROJECT_DIR", "/data")))
store.restore()

import api  # noqa: E402 — project construction must follow restore

app = install(api, store)
