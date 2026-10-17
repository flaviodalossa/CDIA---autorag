"""Validated native transfers of prepared bases; never invoke a text/model provider.

A base import upserts documents and keeps the destination profiles and active
profile. Source embedding controls are discarded and rebuilt from destination
manifests. Optional legacy NPZ vectors are published in their own OpenAI profile.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from .core import CATEGORIES, digest, documents, full_topic, normalize, topics, write_json
from .service import legacy_profile, legacy_record, sync_base
from .web_service import merge_documents

MAX_TRANSFER_BYTES = 25 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
DIGEST_ALGORITHM = "sha256.documents_without_embedding_controls.v1"
_SECRET_KEY = re.compile(r"^(?:api[_-]?key|api[_-]?token|access[_-]?token|password|senha|secret|authorization|credentials?)$", re.I)
_ID = re.compile(r"[\w:.-]{1,512}", re.UNICODE)
_REMOVED = {"embedding", "embeddings", "controle_embeddings"}


def _without_controls(rows):
    rows = deepcopy(rows)
    if not isinstance(rows, list):
        return rows
    for document in rows:
        if not isinstance(document, dict):
            continue
        for key in _REMOVED:
            document.pop(key, None)
        structure = document.get("estrutura", [])
        if isinstance(structure, list):
            for topic in structure:
                if isinstance(topic, dict):
                    for key in _REMOVED:
                        topic.pop(key, None)
    return rows


def canonical_digest(base):
    """Digest document content independent of index controls and root revisions."""
    rows = _without_controls(documents(base))
    rows.sort(key=lambda row: row["documento_id"])
    return digest({"documentos": rows})


def _json_tree(value, depth=0):
    if depth > 64:
        raise ValueError("JSON excede o limite de profundidade.")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Chaves JSON devem ser textos.")
            if _SECRET_KEY.fullmatch(key):
                raise ValueError("Credenciais não podem fazer parte da base importada.")
            _json_tree(item, depth+1)
    elif isinstance(value, list):
        for item in value:
            _json_tree(item, depth+1)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError("O JSON contém número não finito.")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("Valor incompatível com JSON.")


def _identifier(value, field):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{field} deve ser um identificador não vazio, sem caminhos, de até 512 caracteres.")


def _text(value, field, allow_null=False):
    if allow_null and value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{field} deve ser texto.")


def _texts(value, field):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} deve ser uma lista de textos.")


def validate_documents(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2000:
        raise ValueError("A base deve conter uma lista documentos com 1 a 2000 documentos.")
    rows = _without_controls(deepcopy(rows))
    _json_tree(rows)
    document_ids, topic_ids = set(), set()
    for document in rows:
        if not isinstance(document, dict):
            raise ValueError("Cada documento deve ser um objeto JSON.")
        doc_id = document.get("documento_id")
        _identifier(doc_id, "documento_id")
        if doc_id in document_ids:
            raise ValueError("documento_id duplicado na base.")
        document_ids.add(doc_id)
        _text(document.get("nome_documento"), "nome_documento")
        for field in ("identificacao", "metadados", "vigencia", "fonte", "controle_processamento", "validacao_importacao"):
            if field in document and not isinstance(document[field], dict):
                raise ValueError(f"{field} do documento deve ser um objeto.")
        metadata = document.get("metadados", {})
        for field in ("eixos", "eixo"):
            if field in metadata and metadata[field] is not None:
                value = metadata[field]
                if not isinstance(value, str):
                    _texts(value, field)
        for field in ("ativo",):
            if field in document and not isinstance(document[field], bool):
                raise ValueError(f"{field} do documento deve ser booleano.")
        if "ordem_documento" in document and (type(document["ordem_documento"]) is not int or document["ordem_documento"] < 0):
            raise ValueError("ordem_documento deve ser inteiro não negativo.")
        structure = document.get("estrutura")
        if not isinstance(structure, list):
            raise ValueError("estrutura deve ser uma lista de tópicos.")
        for topic in structure:
            if not isinstance(topic, dict):
                raise ValueError("Cada tópico deve ser um objeto.")
            topic_id = topic.get("topico_id")
            _identifier(topic_id, "topico_id")
            if topic_id in topic_ids:
                raise ValueError("topico_id duplicado na base.")
            topic_ids.add(topic_id)
            if len(topic_ids) > 50000:
                raise ValueError("A base excede 50 mil tópicos.")
            for field in ("titulo", "conteudo_fonte"):
                _text(topic.get(field), field)
            for field in ("resumo", "resposta_sintese"):
                if field in topic:
                    _text(topic[field], field)
            for field in ("caminho_hierarquico", "perguntas_sugeridas", "base_normativa", "palavras_chave"):
                if field in topic:
                    _texts(topic[field], field)
            if "secao" in topic:
                _text(topic["secao"], "secao", allow_null=True)
            if "ativo" in topic and not isinstance(topic["ativo"], bool):
                raise ValueError("ativo do tópico deve ser booleano.")
            if "ordem_no_documento" in topic and (type(topic["ordem_no_documento"]) is not int or topic["ordem_no_documento"] < 0):
                raise ValueError("ordem_no_documento deve ser inteiro não negativo.")
            for field in ("fonte", "controle_topico", "localizacao", "processamento"):
                if field in topic and not isinstance(topic[field], dict):
                    raise ValueError(f"{field} do tópico deve ser um objeto.")
    return rows


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("O JSON contém chave duplicada.")
        result[key] = value
    return result


@dataclass(frozen=True)
class LegacyVectors:
    model: str
    dimensions: int
    ids: tuple[str, ...]
    hashes: tuple[str, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class PreparedBase:
    documents: list[dict]
    canonical_digest: str
    legacy: LegacyVectors | None = None
    base_id: str | None = None
    root_metadata: dict = field(default_factory=dict)


def _legacy_vectors(data, rows):
    allowed = {"vetores.npy", "topico_ids.npy", "hashes_texto.npy", "modelo.npy", "dimensoes.npy", "normalizacao.npy"}
    required = allowed - {"normalizacao.npy"}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            infos = package.infolist()
            names = [entry.filename for entry in infos]
            if len(names) != len(set(names)) or not required <= set(names) or not set(names) <= allowed:
                raise ValueError("NPZ legado deve conter apenas os arrays do formato legado, sem caminhos.")
            if any(entry.flag_bits & 1 for entry in infos):
                raise ValueError("NPZ criptografado não é aceito.")
            if sum(entry.file_size for entry in infos) > MAX_EXPANDED_BYTES:
                raise ValueError("NPZ excede 64 MiB descompactados.")
            # Validate NPY header allocation before NumPy sees attacker-controlled shapes.
            for entry in infos:
                with package.open(entry) as header_stream:
                    version = np.lib.format.read_magic(header_stream)
                    if version == (1, 0):
                        shape, order, dtype = np.lib.format.read_array_header_1_0(header_stream)
                    elif version == (2, 0):
                        shape, order, dtype = np.lib.format.read_array_header_2_0(header_stream)
                    else:
                        raise ValueError("Versão NPY não suportada.")
                    if dtype.hasobject or len(shape) > 2 or dtype.itemsize <= 0:
                        raise ValueError("NPZ com objetos ou forma de array insegura.")
                    allocation = math.prod(shape) * dtype.itemsize
                    if allocation > MAX_EXPANDED_BYTES or allocation != entry.file_size-header_stream.tell():
                        raise ValueError("Forma NPY diverge do tamanho real do array.")
                # Check CRC and actual decompressed lengths before loading arrays.
                size = 0
                with package.open(entry) as stream:
                    for block in iter(lambda: stream.read(65536), b""):
                        size += len(block)
                        if size > entry.file_size or size > MAX_EXPANDED_BYTES:
                            raise ValueError("NPZ excede o limite de descompactação.")
        with np.load(io.BytesIO(data), allow_pickle=False) as package:
            values = {key: package[key] for key in package.files}
        matrix = np.asarray(values["vetores"])
        dim = values["dimensoes"]
        if dim.size != 1 or dim.dtype.kind not in "iu":
            raise ValueError("dimensoes do NPZ deve ser um inteiro único.")
        dimensions = int(dim.reshape(-1)[0])
        if not 1 <= dimensions <= 8192:
            raise ValueError("Dimensão do NPZ fora dos limites.")
        model_array = values["modelo"]
        if model_array.size != 1 or model_array.dtype.kind not in "US":
            raise ValueError("modelo do NPZ deve ser texto único.")
        model = str(model_array.reshape(-1)[0])
        if model not in {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}:
            raise ValueError("O NPZ legado deve identificar um modelo OpenAI de embeddings conhecido.")
        if matrix.dtype.kind not in "fiu" or matrix.ndim != 2 or not 1 <= len(matrix) <= 50000:
            raise ValueError("Matriz de embeddings do NPZ inválida.")
        with np.errstate(over="ignore", invalid="ignore"):
            normalize(matrix, dimensions)
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError("NPZ contém norma vetorial inválida.")
        for field in ("topico_ids", "hashes_texto"):
            if values[field].ndim != 1 or values[field].dtype.kind not in "US":
                raise ValueError(f"{field} do NPZ deve ser vetor de textos.")
        ids = tuple(str(value) for value in values["topico_ids"].tolist())
        hashes = tuple(str(value) for value in values["hashes_texto"].tolist())
        if len(ids) != len(matrix) or len(hashes) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("IDs, hashes ou quantidade de vetores do NPZ divergem.")
        mapping = {topic["topico_id"]: topic for doc in rows for topic in doc["estrutura"] if topic.get("ativo", True)}
        if set(ids) != set(mapping):
            raise ValueError("O NPZ deve representar exatamente todos os tópicos ativos da base enviada.")
        if any(not re.fullmatch(r"[a-f0-9]{64}", value) or digest(full_topic(mapping[topic_id])) != value
               for topic_id, value in zip(ids, hashes)):
            raise ValueError("Hash do NPZ difere do conteúdo atual do tópico.")
        if "normalizacao" in values:
            norm = values["normalizacao"]
            if norm.size != 1 or norm.dtype.kind not in "US" or str(norm.reshape(-1)[0]) != "l2":
                raise ValueError("Normalização do NPZ deve ser l2.")
        matrix = matrix.copy()
        matrix.flags.writeable = False
        return LegacyVectors(model, dimensions, ids, hashes, matrix)
    except ValueError:
        raise
    except (zipfile.BadZipFile, OSError, KeyError, RuntimeError, EOFError, OverflowError) as exc:
        raise ValueError("Arquivo NPZ legado inválido ou inseguro.") from exc



def _root_metadata(base, rows):
    _json_tree(base)
    result = {}
    doc_ids = {doc["documento_id"] for doc in rows}
    if "base_id" in base:
        if not isinstance(base["base_id"], str) or len(base["base_id"]) > 300:
            raise ValueError("base_id deve ser um texto de até 300 caracteres.")
        result["base_id"] = base["base_id"]
    if "ordem_documentos" in base:
        order = base["ordem_documentos"]
        _texts(order, "ordem_documentos")
        if len(order) != len(set(order)) or set(order) != doc_ids:
            raise ValueError("ordem_documentos deve identificar exatamente os documentos enviados.")
        result["ordem_documentos"] = deepcopy(order)
    if "eixos" in base:
        axes = base["eixos"]
        if not isinstance(axes, list) or len(axes) > 2000:
            raise ValueError("eixos deve ser lista de até 2000 classificações.")
        seen = set()
        for axis in axes:
            if not isinstance(axis, dict) or not set(axis) <= {"eixo_id", "nome_eixo", "nome", "documentos_ids"}:
                raise ValueError("Cada eixo deve conter somente identidade, nome e referências de documentos.")
            axis_id = axis.get("eixo_id")
            _identifier(axis_id, "eixo_id")
            if axis_id in seen:
                raise ValueError("eixo_id duplicado.")
            seen.add(axis_id)
            _text(axis.get("nome_eixo", axis.get("nome")), "nome_eixo")
            ids = axis.get("documentos_ids")
            _texts(ids, "documentos_ids")
            if len(ids) != len(set(ids)) or not set(ids) <= doc_ids:
                raise ValueError("Eixo referencia documento ausente ou repetido.")
        result["eixos"] = deepcopy(axes)
    if "execucoes_processamento" in base:
        runs = base["execucoes_processamento"]
        allowed = {"execucao_id", "tipo", "inicio", "fim", "status", "modelo", "filtro_documentos_ids", "resultados", "resumo"}
        if not isinstance(runs, list) or len(runs) > 10000:
            raise ValueError("Histórico de processamento deve ser uma lista limitada.")
        seen = set()
        for run in runs:
            if not isinstance(run, dict) or not set(run) <= allowed:
                raise ValueError("Campos de histórico de processamento não suportados.")
            _identifier(run.get("execucao_id"), "execucao_id")
            if run["execucao_id"] in seen:
                raise ValueError("execucao_id histórico duplicado.")
            seen.add(run["execucao_id"])
            for key in ("tipo", "inicio", "fim", "status", "modelo"):
                if key in run:
                    _text(run[key], key, allow_null=key == "fim")
            if "filtro_documentos_ids" in run:
                _texts(run["filtro_documentos_ids"], "filtro_documentos_ids")
            if "resultados" in run and (not isinstance(run["resultados"], list) or any(not isinstance(v, dict) for v in run["resultados"])):
                raise ValueError("resultados históricos devem ser uma lista de objetos.")
            if "resumo" in run and not isinstance(run["resumo"], dict):
                raise ValueError("resumo histórico deve ser um objeto.")
        result["execucoes_processamento"] = deepcopy(runs)
    if "estatisticas" in base:
        stats = base["estatisticas"]
        if not isinstance(stats, dict) or any(type(v) is not int or v < 0 for v in stats.values()):
            raise ValueError("estatisticas deve conter contagens inteiras não negativas.")
    return result


def _statistics(base, profile_id):
    active = [topic for topic, _ in topics(base)]
    stats = {"documentos": len(documents(base)), "topicos_ativos": len(active),
             "topicos_inativos": sum(not t.get("ativo", True) for t, _ in topics(base, active=False)),
             "embeddings_pendentes": 0, "embeddings_desatualizados": 0, "embeddings_atualizados": 0}
    for topic in active:
        categories = topic.get("embeddings", {}).get("perfis", {}).get(profile_id, {}).get("categorias", {})
        states = {category.get("status") for category in categories.values()}
        field = "embeddings_desatualizados" if "desatualizado" in states else "embeddings_atualizados" if states and states == {"atualizado"} else "embeddings_pendentes"
        stats[field] += 1
    return stats

def prepare_base_import(base_data: bytes, legacy_npz: bytes | None = None):
    """Parse and validate the entire transfer without writing any project file."""
    if not isinstance(base_data, bytes) or not base_data:
        raise ValueError("Forneça base_file como arquivo JSON não vazio.")
    if legacy_npz is not None and not isinstance(legacy_npz, bytes):
        raise ValueError("legacy_npz deve ser um arquivo NPZ.")
    if len(base_data) + len(legacy_npz or b"") > MAX_TRANSFER_BYTES:
        raise ValueError("A transferência excede 25 MiB.")
    try:
        base = json.loads(base_data.decode("utf-8-sig"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("JSON contém número não finito.")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("base_file deve conter JSON UTF-8 válido.") from exc
    if not isinstance(base, dict):
        raise ValueError("A raiz da base deve ser um objeto JSON.")
    _json_tree(base)
    # Only documents are imported: client profiles, server paths and credentials
    # in index controls are not trusted or used.
    if "base_id" in base and (not isinstance(base["base_id"], str) or len(base["base_id"]) > 300):
        raise ValueError("base_id deve ser um texto de até 300 caracteres.")
    rows = validate_documents(base.get("documentos"))
    legacy = _legacy_vectors(legacy_npz, rows) if legacy_npz is not None else None
    return PreparedBase(rows, canonical_digest({"documentos": rows}), legacy, base.get("base_id"), _root_metadata(base, rows))


def _destination(project, incoming):
    old = project.base() if project.base_path.exists() else {"documentos": []}
    merged = {doc["documento_id"]: doc for doc in documents(old)}
    merged.update({doc["documento_id"]: doc for doc in incoming})
    validate_documents(list(merged.values()))
    config = project.config()
    for profile_id in config.get("perfis", {}):
        for category in CATEGORIES:
            project.load_index(profile_id, category)
    return config


class _AlreadyLocked:
    def __init__(self, project):
        self._project = project

    def __getattr__(self, name):
        return getattr(self._project, name)

    @contextmanager
    def lock(self):
        yield


def import_prepared_base(project, prepared):
    """Revalidate destination and upsert under one project lock; preserve profiles."""
    incoming = validate_documents(prepared.documents)
    if canonical_digest({"documentos": incoming}) != prepared.canonical_digest:
        raise ValueError("A base preparada foi alterada antes da importação.")
    root_metadata = _root_metadata(prepared.root_metadata, incoming)
    legacy = prepared.legacy
    if legacy:
        active = {t["topico_id"]: t for d in incoming for t in d["estrutura"] if t.get("ativo", True)}
        if (not 1 <= legacy.dimensions <= 8192 or legacy.matrix.ndim != 2
                or legacy.matrix.shape != (len(legacy.ids), legacy.dimensions)
                or legacy.matrix.dtype.kind not in "fiu"
                or len(set(legacy.ids)) != len(legacy.ids) or set(legacy.ids) != set(active)
                or len(legacy.hashes) != len(legacy.ids)
                or legacy.model not in {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
                or any(digest(full_topic(active[i])) != h for i, h in zip(legacy.ids, legacy.hashes))):
            raise ValueError("Vetores preparados divergem da base atual.")
        with np.errstate(over="ignore", invalid="ignore"):
            normalize(legacy.matrix, legacy.dimensions)
            norms = np.linalg.norm(legacy.matrix.astype(np.float64), axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError("Vetores preparados possuem normas inválidas.")
    profile = legacy_profile(legacy.model, legacy.dimensions) if legacy else None
    with project.lock():
        config = _destination(project, incoming)
        previous_active = config.get("perfil_ativo_id")
        previous_base = project.base() if project.base_path.exists() else {"documentos": []}
        destination_empty = not documents(previous_base)
        if profile and profile["perfil_id"] in config["perfis"] and config["perfis"][profile["perfil_id"]] != profile:
            raise ValueError("O perfil legado existente diverge da identidade do NPZ.")
        if profile:
            project.load_index(profile["perfil_id"], CATEGORIES[0])
        paths = [project.base_path, project.config_path]
        if profile:
            paths.append(project.root/"indices"/profile["perfil_id"]/CATEGORIES[0]/"atual.json")
        snapshots = {path: json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None for path in paths}
        locked = _AlreadyLocked(project)
        try:
            merge_documents(locked, incoming)
            if legacy:
                current_base = project.base()
                mapping = {topic["topico_id"]: (topic, doc) for topic, doc in topics(current_base, active=False)}
                records, vectors = [], []
                stored = project.load_index(profile["perfil_id"], CATEGORIES[0])
                incoming_docs = {doc["documento_id"] for doc in incoming}
                previous_topic_documents = {t["topico_id"]: d["documento_id"] for t, d in topics(previous_base, active=False)}
                if stored:
                    for record, vector in zip(stored[0]["itens"], stored[1]):
                        old_document = record.get("documento_id") or previous_topic_documents.get(record["topico_id"])
                        if old_document not in incoming_docs:
                            records.append(deepcopy(record))
                            vectors.append(vector)
                for topic_id, hash_value, vector in zip(legacy.ids, legacy.hashes, legacy.matrix):
                    topic, document = mapping[topic_id]
                    record = legacy_record(topic, full_topic(topic), current_base, recorded_hash=hash_value)
                    record["documento_id"] = document["documento_id"]
                    records.append(record)
                    vectors.append(vector)
                config = project.config()
                config.setdefault("perfis", {})[profile["perfil_id"]] = profile
                config["perfil_ativo_id"] = previous_active
                project.save_config(config)
                project.publish(profile, CATEGORIES[0], records, np.stack(vectors))
                sync_base(locked)
            updated = project.base()
            if destination_empty:
                updated.update(deepcopy(root_metadata))
            else:
                # Keep the destination's established taxonomy/provenance. Only
                # append new document references, without renaming existing axes.
                old_ids = {doc["documento_id"] for doc in documents(previous_base)}
                new_ids = {doc["documento_id"] for doc in documents(updated)} - old_ids
                generated_axes = updated.get("eixos", [])
                for key in ("base_id", "ordem_documentos", "eixos", "execucoes_processamento"):
                    if key in previous_base:
                        updated[key] = deepcopy(previous_base[key])
                if "ordem_documentos" in previous_base:
                    updated["ordem_documentos"].extend(doc["documento_id"] for doc in documents(updated) if doc["documento_id"] in new_ids)
                if "eixos" in previous_base:
                    axes = updated["eixos"]
                    by_name = {axis.get("nome_eixo", axis.get("nome")): axis for axis in axes}
                    for axis in generated_axes:
                        additions = [doc_id for doc_id in axis.get("documentos_ids", []) if doc_id in new_ids]
                        if not additions:
                            continue
                        name = axis.get("nome_eixo", axis.get("nome"))
                        if name in by_name:
                            by_name[name]["documentos_ids"].extend(additions)
                        else:
                            axes.append({**axis, "documentos_ids": additions})
                            by_name[name] = axes[-1]
            updated["estatisticas"] = _statistics(updated, project.config().get("perfil_ativo_id"))
            write_json(project.base_path, updated)
        except Exception:
            # New immutable revisions may remain unreferenced; restore all active
            # pointers, config and base before the failed job is checkpointed.
            for path, old in reversed(list(snapshots.items())):
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    write_json(path, old)
            raise
        base = project.base()
        incoming_topics = [topic for document in incoming for topic in document["estrutura"]]
        total_topics = list(topics(base, active=False))
        return {
            "documents": len(incoming), "topics": len(incoming_topics),
            "active_topics": sum(t.get("ativo", True) for t in incoming_topics),
            "inactive_topics": sum(not t.get("ativo", True) for t in incoming_topics),
            "total_documents": len(documents(base)), "total_topics": len(total_topics),
            "canonical_digest": prepared.canonical_digest,
            "base_canonical_digest": canonical_digest(base), "digest_algorithm": DIGEST_ALGORITHM,
            "legacy_embeddings": len(legacy.ids) if legacy else 0,
            "legacy_profile_id": profile["perfil_id"] if profile else None,
            "active_profile_id": project.config().get("perfil_ativo_id"),
            "root_metadata_preserved": destination_empty,
            "root_metadata_fields": sorted(prepared.root_metadata) if destination_empty else [],
            "committed": True,
        }
