"""Offline image smoke test: real embeddings, no R2, model downloads, or paid APIs."""
from __future__ import annotations

import argparse
import os

import numpy as np

from .boot import OLLAMA_URL, local_json, start_ollama, stop_processes
from .download_models import verify_installation


def verify(smoke=False):
    lock = verify_installation(os.environ.get("OLLAMA_MODELS", "/opt/models/ollama"), full=True)
    if not smoke:
        print("Pinned manifests, all blob SHA256 checksums and license layers verified.")
        return
    from autorag.core import normalize
    from autorag.providers import get_provider, profile_spec

    process = start_ollama()
    try:
        for model in lock["models"]:
            # Verify native model output separately from the application's configured reduction.
            native = local_json("/api/embed", {"model": model["name"], "input": ["Consentimento livre e esclarecido."],
                "truncate": False, "keep_alive": 0, "options": {"num_thread": 1}}, timeout=600)
            coordinates = np.asarray(native["embeddings"], dtype=np.float64)
            normalize(coordinates, model["native_dimensions"])
            profile = profile_spec("ollama", model["name"], model["dimensions"], endpoint=OLLAMA_URL)
            if profile["revisao_modelo"] != model["manifest_sha256"]:
                raise ValueError("Application profile model digest mismatch.")
            values = get_provider(profile).embed(["Qual é a exigência normativa?"], "consulta")
            normalize(values, model["dimensions"])
            if values.shape != (1, model["dimensions"]):
                raise ValueError("Unexpected application embedding shape.")
            print("Offline CPU embedding passed:", model["name"], "native", model["native_dimensions"], "profile", model["dimensions"], flush=True)
    finally:
        stop_processes([process])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    verify(parser.parse_args().smoke)
