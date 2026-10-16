from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np


CATEGORIES = (
    "topico_total_sem_palavras_chave", "topico_total_com_palavras_chave",
    "resumo", "conteudo_fonte", "perguntas_sugeridas", "resposta_sintese",
    "palavras_chave", "metadados",
)
LABELS = {
    "topico_total_sem_palavras_chave": "Tópico total sem palavras-chave",
    "topico_total_com_palavras_chave": "Tópico total com palavras-chave",
    "resumo": "Resumo", "conteudo_fonte": "Conteúdo-fonte",
    "perguntas_sugeridas": "Perguntas sugeridas", "resposta_sintese": "Resposta-síntese",
    "palavras_chave": "Palavras-chave", "metadados": "Metadados",
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def uid(prefix):
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:12]}"


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".gravar_", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Windows scanners/readers can briefly hold the destination open.
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.025 * (attempt + 1))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".vetores_", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalize(values, dimensions=None):
    values = np.asarray(values)
    if values.dtype.kind == "c":
        raise ValueError("Embeddings devem ter coordenadas reais.")
    if values.dtype.kind not in "fc":
        values = values.astype(np.float64)
    if values.ndim != 2 or (dimensions is not None and values.shape[1] != dimensions):
        raise ValueError(f"Formato de vetores inválido: {values.shape}; dimensão esperada: {dimensions}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Vetor contém valores não finitos.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Vetor nulo não pode ser normalizado.")
    return values / norms


def text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(text(item) for item in value if text(item))
    return str(value).strip()


def full_topic(topic):
    # Composição legada preservada, incluindo quebras de linha das listas.
    return (
        f"Título: {topic.get('titulo', '')}\n"
        f"Caminho: {' > '.join(topic.get('caminho_hierarquico', []))}\n"
        f"Resumo: {topic.get('resumo', '')}\n"
        f"Conteúdo: {topic.get('conteudo_fonte', '')}\n"
        f"Perguntas: {chr(10).join(topic.get('perguntas_sugeridas', []))}\n"
        f"Resposta: {topic.get('resposta_sintese', '')}\n"
        f"Base normativa: {'; '.join(topic.get('base_normativa', []))}"
    ).strip()


def category_text(topic, document, category):
    if category not in CATEGORIES:
        raise ValueError(f"Categoria desconhecida: {category}")
    if category == CATEGORIES[0]:
        return full_topic(topic)
    if category == CATEGORIES[1]:
        return full_topic(topic) + "\nPalavras-chave: " + text(topic.get("palavras_chave"))
    if category == "metadados":
        identity = document.get("identificacao", {})
        meta = document.get("metadados", {})
        validity = document.get("vigencia", {})
        fields = [("Documento", document.get("nome_documento"))]
        fields += [(name, identity.get(name)) for name in (
            "tipo_documento", "identificacao_documento", "data_documento", "descricao_documento")]
        fields += [("eixos", meta.get("eixos") or meta.get("eixo")), ("idioma", meta.get("idioma"))]
        fields += [(name, validity.get(name)) for name in ("inicio_vigencia", "fim_vigencia")]
        return "\n".join(f"{name}: {text(value)}" for name, value in fields if text(value))
    value = topic.get(category)
    if category == "perguntas_sugeridas" and isinstance(value, list):
        return "\n".join(text(item) for item in value if text(item))
    return text(value)


def documents(base):
    if isinstance(base.get("documentos"), list):
        return base["documentos"]
    seen = {}
    for axis in base.get("eixos", []):
        for document in axis.get("documentos", []):
            seen.setdefault(document["documento_id"], document)
    return list(seen.values())


def topics(base, active=True):
    seen = set()
    for document in documents(base):
        for topic in document.get("estrutura", []):
            topic_id = topic.get("topico_id")
            if not topic_id or topic_id in seen:
                raise ValueError(f"ID de tópico ausente ou duplicado: {topic_id}")
            seen.add(topic_id)
            if active and not topic.get("ativo", True):
                continue
            yield topic, document


def split_text(value, budget):
    """Limite conservador por bytes UTF-8 para provedores sem tokenizador local.

    Preserva todo o texto e offsets. O provedor também recebe truncate=False.
    """
    if budget < 16:
        raise ValueError("Limite de entrada insuficiente para a preparação do modelo.")
    start = 0
    while start < len(value):
        stop, size = start, 0
        while stop < len(value):
            size += len(value[stop].encode("utf-8"))
            if size > budget:
                break
            stop += 1
        if stop < len(value):
            boundary = value.rfind(" ", start + max(1, (stop-start)//2), stop)
            if boundary > start:
                stop = boundary + 1
        if stop == start:
            raise ValueError("Não foi possível segmentar o texto.")
        yield value[start:stop], start, stop
        start = stop


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = self.root / "dados"
        self.data.mkdir(exist_ok=True)
        self.base_path = self.data / "base_etica.json"
        self.config_path = self.data / "projeto.json"

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Referência de artefato deve ficar dentro do projeto.")
        return path

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def config(self):
        return read_json(self.config_path) if self.config_path.exists() else {
            "schema_version": "1.0", "perfis": {}, "perfil_ativo_id": None}

    def save_config(self, config):
        write_json(self.config_path, config)

    def base(self):
        if not self.base_path.exists():
            raise ValueError("Importe uma base JSON na aba Base ou pelo comando importar.")
        return read_json(self.base_path)

    @contextlib.contextmanager
    def lock(self):
        path = self.data / ".escrita.lock"
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError("Há outra operação em andamento. Se um processo foi interrompido, feche-o e remova dados/.escrita.lock.") from exc
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(str(os.getpid()))
            yield
        finally:
            path.unlink(missing_ok=True)

    def current(self, profile_id, category):
        path = self.root / "indices" / profile_id / category / "atual.json"
        return read_json(path) if path.exists() else None

    def load_index(self, profile_id, category):
        pointer = self.current(profile_id, category)
        if not pointer:
            return None
        manifest = read_json(self.path(pointer["manifesto"]))
        npz_path = self.path(manifest["arquivo_npz"])
        if file_hash(npz_path) != manifest["npz_sha256"]:
            raise ValueError("Checksum do NPZ diverge do manifesto.")
        with np.load(npz_path, allow_pickle=False) as package:
            matrix = normalize(package["vetores"], manifest["dimensoes"])
            ids = package["embedding_ids"].tolist()
            original = package["vetores_originais"].copy() if "vetores_originais" in package else None
        if ids != [item["embedding_id"] for item in manifest["itens"]] or len(set(ids)) != len(ids):
            raise ValueError("IDs ou ordem do NPZ divergentes do manifesto.")
        if len(matrix) != len(manifest["itens"]) or (original is not None and (original.shape != matrix.shape or not np.all(np.isfinite(original)))):
            raise ValueError("Tamanho da matriz divergente do manifesto.")
        return manifest, matrix, original

    def publish(self, profile, category, items, matrix, original=None):
        matrix = normalize(matrix, profile["dimensoes"])
        if len(items) != len(matrix):
            raise ValueError("Quantidade de unidades difere da matriz.")
        revision = uid("rev")
        folder = self.root / "indices" / profile["perfil_id"] / category / revision
        ids = [item["embedding_id"] for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Unidades de embedding duplicadas.")
        npz_path = folder / "vetores.npz"
        arrays = {"vetores": matrix, "embedding_ids": np.asarray(ids, dtype=np.str_)}
        if original is not None:
            if np.asarray(original).shape != matrix.shape or not np.all(np.isfinite(original)):
                raise ValueError("Vetores originais inválidos.")
            arrays["vetores_originais"] = np.asarray(original)
        write_npz(npz_path, **arrays)
        manifest = {
            "schema_version": "2.0", "perfil_id": profile["perfil_id"], "perfil": profile,
            "categoria": category, "revisao_indice": revision, "data": now(),
            "arquivo_npz": self.relative(npz_path), "npz_sha256": file_hash(npz_path),
            "dimensoes": profile["dimensoes"], "normalizacao": "l2",
            "total_embeddings": len(items), "itens": items,
        }
        for row, item in enumerate(items):
            item["referencia"] = {"arquivo_npz": self.relative(npz_path), "revisao_indice": revision,
                                  "linha": row, "embedding_id": item["embedding_id"]}
        manifest_path = folder / "manifesto.json"
        write_json(manifest_path, manifest)
        # Somente agora altera a referência corrente; versões anteriores são imutáveis.
        write_json(folder.parent / "atual.json", {"manifesto": self.relative(manifest_path)})
        return manifest
