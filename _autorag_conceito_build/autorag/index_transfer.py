"""Bounded, model-free transfer of one native index into an existing profile."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import re
import stat
import zipfile

import numpy as np

from .core import CATEGORIES, category_text, digest, normalize, topics, write_json
from .providers import Provider
from .service import sync_base

MAX_ZIP_BYTES = 25 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_FILES = {"transfer.json", "manifesto.json", "vetores.npz"}
_PROFILE_FIELDS = (
    "provedor", "modelo", "dimensoes", "dimensoes_nativas", "limite_tokens",
    "revisao_modelo", "normalizacao", "versao_preparacao", "segmentacao",
    "reducao", "legado",
)
_RECORD_FIELDS = {
    "embedding_id", "topico_id", "documento_id", "categoria", "segmento",
    "inicio", "fim", "hash_texto_fonte", "hash_texto_embeddado", "texto_segmento",
    "data_geracao", "revisao_base_geracao", "original_disponivel", "referencia",
}
_ERROR = "Pacote de índice inválido ou incompatível com o projeto."


@dataclass
class _Prepared:
    profile: dict
    category: str
    records: list
    matrix: np.ndarray
    originals: np.ndarray | None
    checksum: str


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(_ERROR)
            result[key] = value
        return result

    def constant(_):
        raise ValueError(_ERROR)

    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                       parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError(_ERROR)
    return value


def _members(archive, names, limit):
    infos = archive.infolist()
    if len(infos) != len(names) or {item.filename for item in infos} != names:
        raise ValueError(_ERROR)
    total = 0
    for item in infos:
        mode = stat.S_IFMT(item.external_attr >> 16)
        if (item.is_dir() or item.flag_bits & 1 or item.external_attr & 0x10
                or mode not in (0, stat.S_IFREG)
                or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or item.file_size < 0):
            raise ValueError(_ERROR)
        total += item.file_size
        if total > limit:
            raise ValueError(_ERROR)
    return infos


def _read_members(data, names, limit):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = _members(archive, names, limit)
        result = {}
        for item in infos:
            with archive.open(item) as stream:
                value = stream.read(item.file_size + 1)
            if len(value) != item.file_size:
                raise ValueError(_ERROR)
            result[item.filename] = value
        return result


def _profile_spec(profile):
    if not isinstance(profile, dict):
        raise ValueError(_ERROR)
    spec = {key: deepcopy(profile.get(key)) for key in _PROFILE_FIELDS}
    if (any(not isinstance(spec[key], str) or not spec[key]
            for key in ("provedor", "modelo", "revisao_modelo"))
            or not _integer(spec["dimensoes"], 1) or spec["dimensoes"] > 8192
            or not _integer(spec["limite_tokens"], 16)
            or spec["normalizacao"] != "l2" or spec["versao_preparacao"] != "1"
            or spec["segmentacao"] != "utf8_conservador_v1"):
        raise ValueError(_ERROR)
    if spec["dimensoes_nativas"] is not None and not _integer(spec["dimensoes_nativas"], 1):
        raise ValueError(_ERROR)
    if spec["legado"] is not None and not isinstance(spec["legado"], bool):
        raise ValueError(_ERROR)
    return spec


def _arrays(data, dimensions, count, remaining):
    # NPZ is another ZIP. Check its expanded sizes AND each NPY header before
    # numpy allocates from a potentially hostile shape or object dtype.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = {item.filename for item in archive.infolist()}
    allowed = {"vetores.npy", "embedding_ids.npy", "vetores_originais.npy"}
    if not {"vetores.npy", "embedding_ids.npy"} <= names or not names <= allowed:
        raise ValueError(_ERROR)
    contents = _read_members(data, names, remaining)
    for name, value in contents.items():
        stream = io.BytesIO(value)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(_ERROR)
        if (dtype.hasobject or dtype.fields is not None
                or any(not _integer(size) for size in shape)
                or math.prod(shape) * dtype.itemsize != len(value) - stream.tell()):
            raise ValueError(_ERROR)
        if name == "embedding_ids.npy":
            if shape != (count,) or dtype.kind not in "US" or dtype.itemsize > 4096:
                raise ValueError(_ERROR)
        elif shape != (count, dimensions) or dtype.kind not in "fiu" or dtype.itemsize > 8:
            raise ValueError(_ERROR)
    del contents
    with np.load(io.BytesIO(data), allow_pickle=False) as package:
        matrix = package["vetores"].copy()
        ids = package["embedding_ids"].tolist()
        originals = package["vetores_originais"].copy() if "vetores_originais" in package else None
    for values in (matrix, originals):
        if values is not None:
            normalize(values.astype(np.float64), dimensions)
            norms = np.linalg.norm(values.astype(np.float64), axis=1)
            if not np.all(np.isfinite(norms)) or np.any(norms == 0):
                raise ValueError(_ERROR)
    return matrix, ids, originals


def _prepare(project, data):
    if not isinstance(data, (bytes, bytearray)) or not 0 < len(data) <= MAX_ZIP_BYTES:
        raise ValueError(_ERROR)
    files = _read_members(data, _FILES, MAX_UNCOMPRESSED_BYTES)
    transfer, manifest = _json(files["transfer.json"]), _json(files["manifesto.json"])
    if set(transfer) != {"profile_id", "category"}:
        raise ValueError(_ERROR)
    profile_id, category = transfer["profile_id"], transfer["category"]
    if (not isinstance(profile_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", profile_id)
            or category not in CATEGORIES):
        raise ValueError(_ERROR)
    profile = project.config().get("perfis", {}).get(profile_id)
    expected_spec = _profile_spec(profile)
    source_spec = manifest.get("perfil")
    if (not isinstance(source_spec, dict) or set(source_spec) != set(_PROFILE_FIELDS)
            or digest(_profile_spec(source_spec)) != digest(expected_spec)
            or manifest.get("perfil_id") != profile_id or profile.get("perfil_id") != profile_id
            or manifest.get("categoria") != category or manifest.get("schema_version") != "2.0"
            or manifest.get("arquivo_npz") != "vetores.npz"
            or manifest.get("normalizacao") != "l2"
            or not _integer(manifest.get("dimensoes"), 1)
            or manifest["dimensoes"] != profile["dimensoes"]):
        raise ValueError(_ERROR)
    records = manifest.get("itens")
    if (not isinstance(records, list)
            or not _integer(manifest.get("total_embeddings"))
            or manifest["total_embeddings"] != len(records)):
        raise ValueError(_ERROR)
    checksum = hashlib.sha256(files["vetores.npz"]).hexdigest()
    if manifest.get("npz_sha256") != checksum:
        raise ValueError(_ERROR)
    remaining = MAX_UNCOMPRESSED_BYTES - len(files["transfer.json"]) - len(files["manifesto.json"])
    matrix, ids, originals = _arrays(files["vetores.npz"], profile["dimensoes"], len(records), remaining)
    if ids != [record.get("embedding_id") for record in records] or len(set(ids)) != len(ids):
        raise ValueError(_ERROR)
    mapping = {topic["topico_id"]: (topic, document) for topic, document in topics(project.base())}
    preparer = Provider(profile)  # Pure preparation; no provider discovery or inference.
    expected, seen = {}, {}
    for record in records:
        if not isinstance(record, dict) or not set(record) <= _RECORD_FIELDS:
            raise ValueError(_ERROR)
        topic_id = record.get("topico_id")
        if not isinstance(topic_id, str) or topic_id not in mapping:
            raise ValueError(_ERROR)
        topic, document = mapping[topic_id]
        if topic_id not in expected:
            value = category_text(topic, document, category)
            old_unit = profile.get("legado") and record.get("original_disponivel") is False
            segments = [(value, 0, len(value))] if old_unit and value else preparer.segments(value)
            expected[topic_id] = (value, segments)
        value, segments = expected[topic_id]
        number, start, end = record.get("segmento"), record.get("inicio"), record.get("fim")
        if (not value or not _integer(number) or number >= len(segments)
                or not _integer(start) or not _integer(end) or end <= start
                or number in seen.setdefault(topic_id, set())):
            raise ValueError(_ERROR)
        segment, actual_start, actual_end = segments[number]
        if (record.get("documento_id") != document["documento_id"]
                or record.get("categoria") != category
                or record.get("embedding_id") != f"{topic_id}::{category}::{number:04d}"
                or record.get("hash_texto_fonte") != digest(value)
                or record.get("texto_segmento") != segment or (start, end) != (actual_start, actual_end)
                or record.get("hash_texto_embeddado") != digest(preparer.prepare(segment, "documento"))
                or not isinstance(record.get("original_disponivel"), bool)
                or any(not isinstance(record.get(key), str) or not record[key]
                       for key in ("data_geracao", "revisao_base_geracao"))
                or record["original_disponivel"] and originals is None):
            raise ValueError(_ERROR)
        seen[topic_id].add(number)
    if any(numbers != set(range(len(expected[topic_id][1]))) for topic_id, numbers in seen.items()):
        raise ValueError(_ERROR)
    # Publishing replaces this category's pointer. A partial source index must
    # never hide otherwise valid destination topics; empty fields need no row.
    eligible = {topic_id for topic_id, pair in mapping.items() if category_text(*pair, category)}
    if set(seen) != eligible:
        raise ValueError(_ERROR)
    clean = [{key: deepcopy(value) for key, value in record.items() if key != "referencia"}
             for record in records]
    return _Prepared(deepcopy(profile), category, clean, matrix, originals, checksum)


def _report(prepared):
    return {"profile_id": prepared.profile["perfil_id"], "category": prepared.category,
            "dimensions": prepared.profile["dimensoes"], "vectors": len(prepared.records),
            "topics": len({row["topico_id"] for row in prepared.records}),
            "documents": len({row["documento_id"] for row in prepared.records}),
            "topic_ids": sorted({row["topico_id"] for row in prepared.records}),
            "embedding_ids": [row["embedding_id"] for row in prepared.records],
            "npz_sha256": prepared.checksum}


def validate_index_bundle(project, data):
    """Validate without writes, model calls, credential lookup or profile creation."""
    try:
        return _report(_prepare(project, data))
    except Exception:
        raise ValueError(_ERROR) from None


def export_index_bundle(project, profile_id, category):
    """Export exactly one current index; server endpoints never enter the bundle."""
    try:
        if category not in CATEGORIES or not isinstance(profile_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", profile_id):
            raise ValueError(_ERROR)
        with project.lock():
            profile = project.config().get("perfis", {}).get(profile_id)
            spec = _profile_spec(profile)
            pointer = project.current(profile_id, category)
            if pointer is None:
                raise ValueError(_ERROR)
            manifest_path = project.path(pointer["manifesto"])
            if manifest_path.stat().st_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(_ERROR)
            source_manifest = _json(manifest_path.read_bytes())
            if digest(_profile_spec(source_manifest.get("perfil"))) != digest(spec):
                raise ValueError(_ERROR)
            path = project.path(source_manifest["arquivo_npz"])
            if path.stat().st_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(_ERROR)
            npz = path.read_bytes()
            manifest = {key: deepcopy(source_manifest[key]) for key in (
                "schema_version", "perfil_id", "categoria", "revisao_indice", "data",
                "npz_sha256", "dimensoes", "normalizacao", "total_embeddings")}
            manifest.update(perfil=spec, arquivo_npz="vetores.npz", itens=[
                {key: deepcopy(value) for key, value in record.items()
                 if key in _RECORD_FIELDS and key != "referencia"}
                for record in source_manifest["itens"]])
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("transfer.json", _json_bytes({"profile_id": profile_id, "category": category}))
                archive.writestr("manifesto.json", _json_bytes(manifest))
                archive.writestr("vetores.npz", npz)
            data = stream.getvalue()
            _prepare(project, data)
            return data
    except Exception:
        raise ValueError(_ERROR) from None


def import_index_bundle(project, data):
    """Revalidate under the project lock, publish a new revision and synchronize."""
    try:
        with project.lock():
            prepared = _prepare(project, data)
            # All existing indexes are checked before sync_base can encounter them.
            for profile_id in project.config().get("perfis", {}):
                for category in CATEGORIES:
                    project.load_index(profile_id, category)
            pointer = project.root / "indices" / prepared.profile["perfil_id"] / prepared.category / "atual.json"
            previous = project.current(prepared.profile["perfil_id"], prepared.category)
            base_before = project.base()
            try:
                published = project.publish(prepared.profile, prepared.category, prepared.records,
                                            prepared.matrix, prepared.originals)
                sync_base(project)
            except Exception:
                if previous is None:
                    pointer.unlink(missing_ok=True)
                else:
                    write_json(pointer, previous)
                write_json(project.base_path, base_before)
                raise
            report = _report(prepared)
            report.update(revision=published["revisao_indice"], npz_sha256=published["npz_sha256"])
            return report
    except Exception:
        raise ValueError(_ERROR) from None
