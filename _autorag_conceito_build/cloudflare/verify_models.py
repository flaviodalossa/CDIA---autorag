"""Offline image smoke test: real embeddings, no R2, model downloads, or paid APIs."""
from __future__ import annotations

import argparse
import os

import numpy as np

from .boot import OLLAMA_URL, local_json, start_ollama, stop_processes
from .download_models import verify_installation
from .download_sentence_models import DEFAULT_ROOT, verify_installation as verify_sentence_installation


def verify(smoke=False):
    lock = verify_installation(os.environ.get("OLLAMA_MODELS", "/opt/models/ollama"), full=True)
    sentence_root = os.environ.get("AUTORAG_MODELS_DIR", DEFAULT_ROOT)
    sentence_models = verify_sentence_installation(sentence_root, full=True)
    if not smoke:
        print("Pinned Ollama and Sentence Transformer assets, SHA256 checksums and model settings verified.")
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

    # Ollama is stopped before loading Torch. The same offline CPU constraints
    # are used by the application; no project or active profile is changed.
    import torch
    if torch.version.cuda is not None:
        raise ValueError("The image must contain the CPU-only Torch build.")
    for model in sentence_models:
        profile = profile_spec("sentence_transformers", model["name"], model["dimensions"], models_dir=sentence_root)
        if profile["limite_tokens"] != model["max_seq_length"]:
            raise ValueError("Sentence Transformer sequence limit differs from its pinned configuration.")
        provider = get_provider(profile)
        if str(provider.model.device) != "cpu":
            raise ValueError("Sentence Transformer must use the CPU.")
        for role, text in [("consulta", "Quais são os direitos do participante?"),
                           ("documento", "O participante pode retirar seu consentimento.")]:
            values = provider.embed([text], role)
            normalize(values, model["dimensions"])
            if values.shape != (1, model["dimensions"]):
                raise ValueError("Unexpected Sentence Transformer embedding shape.")
        print("Offline CPU embedding passed:", model["name"], "profile", model["dimensions"],
              "max_seq_length", model["max_seq_length"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    verify(parser.parse_args().smoke)
