"""Install pinned, offline Sentence Transformer assets without Hub credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import urllib.parse
import urllib.request

from .download_models import LOCK_PATH, sha256_file

DEFAULT_ROOT = "/opt/models/sentence-transformers"
SAFE_PART = re.compile(r"[A-Za-z0-9_.-]+")
SHA256 = re.compile(r"[a-f0-9]{64}")
REQUIRED = {
    "model.safetensors", "config.json", "config_sentence_transformers.json",
    "modules.json", "sentence_bert_config.json", "1_Pooling/config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "sentencepiece.bpe.model", "unigram.json", "README.md",
}


def safe_path(value):
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return (path.as_posix() == value and not path.is_absolute()
            and all(part not in {".", ".."} and SAFE_PART.fullmatch(part) for part in path.parts))


def load_models(lock_path=LOCK_PATH):
    rows = json.loads(Path(lock_path).read_text(encoding="utf-8"))["sentence_transformers"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing pinned Sentence Transformer models.")
    names = set()
    for model in rows:
        name = model.get("name", "")
        if (not safe_path(name) or len(PurePosixPath(name).parts) != 2 or name in names
                or model.get("repository") != name or model.get("license") != "Apache-2.0"
                or not re.fullmatch(r"[a-f0-9]{40}", model.get("revision", ""))
                or model.get("dimensions") != 384 or model.get("max_seq_length") != 128):
            raise ValueError("Invalid pinned Sentence Transformer identity.")
        names.add(name)
        files = model.get("files", [])
        if not isinstance(files, list) or len(files) != len(REQUIRED) or {row.get("path") for row in files} != REQUIRED:
            raise ValueError("The model lock must retain exactly its audited assets and model card.")
        for record in files:
            if (not safe_path(record["path"]) or type(record.get("size")) is not int
                    or not 0 < record["size"] <= (600 if record["path"] == "model.safetensors" else 20) * 1024**2
                    or not SHA256.fullmatch(record.get("sha256", ""))):
                raise ValueError("Invalid Sentence Transformer asset checksum or size.")
    return rows


def model_path(root, model):
    return Path(root) / model["name"].replace("/", "--")


def open_asset(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "AutoRAG-Cloudflare-model-build/1", "Accept": "application/octet-stream",
    }), timeout=120)


def download_asset(root, model, record, opener=open_asset):
    destination = model_path(root, model).joinpath(*PurePosixPath(record["path"]).parts)
    if destination.is_file() and destination.stat().st_size == record["size"] and sha256_file(destination) == record["sha256"]:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    url = ("https://huggingface.co/" + model["repository"] + "/resolve/" + model["revision"]
           + "/" + urllib.parse.quote(record["path"], safe="/"))
    try:
        size = 0
        checksum = hashlib.sha256()
        with opener(url) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(1024**2), b""):
                size += len(block)
                if size > record["size"]:
                    raise ValueError("Sentence Transformer asset exceeds its pinned size.")
                output.write(block)
                checksum.update(block)
        if size != record["size"] or checksum.hexdigest() != record["sha256"]:
            raise ValueError("Sentence Transformer asset checksum differs from its pinned revision.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def verify_settings(folder, model):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    sentence = json.loads((folder / "sentence_bert_config.json").read_text(encoding="utf-8"))
    pooling = json.loads((folder / "1_Pooling/config.json").read_text(encoding="utf-8"))
    modules = json.loads((folder / "modules.json").read_text(encoding="utf-8"))
    if (config.get("hidden_size") != model["dimensions"] or config.get("model_type") != "bert"
            or sentence.get("max_seq_length") != model["max_seq_length"]
            or pooling.get("word_embedding_dimension") != model["dimensions"]
            or pooling.get("pooling_mode_mean_tokens") is not True
            or any(pooling.get(key, False) for key in ("pooling_mode_cls_token", "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken"))
            or not isinstance(modules, list) or len(modules) != 2
            or [(row.get("path"), row.get("type")) for row in modules] != [
                ("", "sentence_transformers.models.Transformer"), ("1_Pooling", "sentence_transformers.models.Pooling")]):
        raise ValueError("The pinned MiniLM architecture, sequence limit or mean pooling differs.")


def verify_installation(root=DEFAULT_ROOT, lock_path=LOCK_PATH, full=False):
    models = load_models(lock_path)
    for model in models:
        folder = model_path(root, model)
        for record in model["files"]:
            path = folder.joinpath(*PurePosixPath(record["path"]).parts)
            if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"] or (full and sha256_file(path) != record["sha256"]):
                raise ValueError("Installed Sentence Transformer asset differs from its lock: " + record["path"])
        verify_settings(folder, model)
    return models


def install_models(root=DEFAULT_ROOT, lock_path=LOCK_PATH, opener=open_asset):
    root = Path(root)
    models = load_models(lock_path)
    for model in models:
        for record in model["files"]:
            for attempt in range(3):
                try:
                    download_asset(root, model, record, opener)
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
        print("Installed pinned Sentence Transformer:", model["name"], model["revision"], flush=True)
    licenses = root / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("licenses") / "APACHE-2.0.txt", licenses / "APACHE-2.0.txt")
    (licenses / "NOTICE.txt").write_text(
        "AutoRAG bundles the original Sentence Transformers multilingual MiniLM model assets and model card.\n"
        "Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, Apache License 2.0.\n"
        "Source: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n"
        "The complete Apache License 2.0 and the original README are retained with this image.\n", encoding="utf-8")
    verify_installation(root, lock_path, full=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_installation(args.root, args.lock, full=True)
    else:
        install_models(args.root, args.lock)
