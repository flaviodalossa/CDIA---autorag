"""Install the two audited Ollama models, validating every manifest and blob."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

LOCK_PATH = Path(__file__).with_name("models.lock.json")
DIGEST = re.compile(r"sha256:([a-f0-9]{64})")
SAFE_PART = re.compile(r"[A-Za-z0-9_.-]+")
MANIFEST_LIMIT = 1024 * 1024


def load_lock(path=LOCK_PATH):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock["registry"] != "https://registry.ollama.ai":
        raise ValueError("Only the official Ollama registry is accepted.")
    for model in lock["models"]:
        if not all(SAFE_PART.fullmatch(part) for part in model["repository"].split("/")) or not SAFE_PART.fullmatch(model["tag"]):
            raise ValueError("Unsafe repository or tag in the model lock.")
        if not DIGEST.fullmatch("sha256:" + model["manifest_sha256"]):
            raise ValueError("Invalid pinned model digest.")
    return lock


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blob_path(root, digest):
    match = DIGEST.fullmatch(digest)
    if not match:
        raise ValueError("Invalid blob digest.")
    return Path(root) / "blobs" / ("sha256-" + match[1])


def manifest_path(root, model):
    return Path(root) / "manifests" / "registry.ollama.ai" / model["repository"] / model["tag"]


def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "AutoRAG-Cloudflare-model-build/1",
        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
    }), timeout=120)


def fetch_manifest(lock, model, opener=open_url):
    prefix = lock["registry"] + "/v2/" + model["repository"] + "/manifests/"
    try:
        response = opener(prefix + "sha256:" + model["manifest_sha256"])
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        # Some registry versions only resolve tags; the byte digest remains mandatory.
        response = opener(prefix + model["tag"])
    with response:
        content = response.read(MANIFEST_LIMIT + 1)
    if len(content) > MANIFEST_LIMIT or hashlib.sha256(content).hexdigest() != model["manifest_sha256"]:
        raise ValueError("The registry manifest differs from the audited local model: " + model["name"])
    manifest = json.loads(content)
    validate_manifest(manifest, model)
    return content, manifest


def validate_manifest(manifest, model):
    if manifest.get("schemaVersion") != 2 or not isinstance(manifest.get("layers"), list):
        raise ValueError("Unsupported Ollama manifest.")
    records = [manifest["config"], *manifest["layers"]]
    for record in records:
        if not DIGEST.fullmatch(record.get("digest", "")) or not isinstance(record.get("size"), int) or not 0 < record["size"] < 2 * 1024**3:
            raise ValueError("Invalid manifest blob.")
    model_layers = [record for record in manifest["layers"] if record.get("mediaType") == "application/vnd.ollama.image.model"]
    if len(model_layers) != 1 or model_layers[0]["size"] != model["model_bytes"]:
        raise ValueError("Unexpected model payload size: " + model["name"])
    if not any(record.get("mediaType") == "application/vnd.ollama.image.license" for record in manifest["layers"]):
        raise ValueError("The pinned model manifest must retain its license layer.")
    return records


def download_blob(lock, model, record, root, opener=open_url):
    target = blob_path(root, record["digest"])
    if target.exists() and target.stat().st_size == record["size"] and sha256_file(target) == record["digest"][7:]:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    url = lock["registry"] + "/v2/" + model["repository"] + "/blobs/" + record["digest"]
    try:
        with opener(url) as response, temporary.open("wb") as output:
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > record["size"]:
                    raise ValueError("Blob exceeds its pinned size.")
                output.write(block)
                digest.update(block)
        if size != record["size"] or digest.hexdigest() != record["digest"][7:]:
            raise ValueError("Blob checksum or size does not match its manifest.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def install_models(root, lock_path=LOCK_PATH, opener=open_url):
    root = Path(root)
    lock = load_lock(lock_path)
    licenses = root / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    for model in lock["models"]:
        content, manifest = fetch_manifest(lock, model, opener)
        for record in validate_manifest(manifest, model):
            for attempt in range(3):
                try:
                    download_blob(lock, model, record, root, opener)
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        destination = manifest_path(root, model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        license_layers = [record for record in manifest["layers"] if record.get("mediaType") == "application/vnd.ollama.image.license"]
        for number, record in enumerate(license_layers, 1):
            name = model["name"].replace(":", "--") + f"--LICENSE-{number}.txt"
            (licenses / name).write_bytes(blob_path(root, record["digest"]).read_bytes())
        print("Installed pinned model:", model["name"], model["manifest_sha256"], flush=True)
    (licenses / "NOTICE.txt").write_text(
        "AutoRAG bundles the original, unmodified Ollama model manifests and license layers.\n"
        "bge-m3: MIT, https://ollama.com/library/bge-m3\n"
        "EmbeddingGemma: Gemma Terms of Use, https://ai.google.dev/gemma/terms\n"
        "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.\n"
        "The original terms bundled with each pinned model are retained alongside this notice.\n",
        encoding="utf-8")
    verify_installation(root, lock_path, full=True)


def verify_installation(root, lock_path=LOCK_PATH, full=False):
    root = Path(root)
    lock = load_lock(lock_path)
    for model in lock["models"]:
        content = manifest_path(root, model).read_bytes()
        if hashlib.sha256(content).hexdigest() != model["manifest_sha256"]:
            raise ValueError("Installed manifest digest mismatch: " + model["name"])
        manifest = json.loads(content)
        for record in validate_manifest(manifest, model):
            path = blob_path(root, record["digest"])
            if path.stat().st_size != record["size"] or (full and sha256_file(path) != record["digest"][7:]):
                raise ValueError("Installed blob checksum or size mismatch: " + model["name"])
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/opt/models/ollama")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_installation(args.root, full=True)
    else:
        install_models(args.root)
