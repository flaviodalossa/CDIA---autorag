"""Durable R2 snapshots through a private Cloudflare container outbound handler."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile

STATE_DIRS = frozenset({"dados", "indices", "historico", "estudos", "previews"})
MAX_SNAPSHOT = 90 * 1024 * 1024
MAX_UNPACKED = 1024 * 1024 * 1024
MAX_JOB = 8 * 1024 * 1024
JOB_ID = re.compile(r"job_[A-Za-z0-9_]+")


def allowed_member(name):
    path = PurePosixPath(name)
    return (path.as_posix() == name and not path.is_absolute() and len(path.parts) > 1 and path.parts[0] in STATE_DIRS
            and all(part not in {".", ".."} and not part.startswith(".")
                    and ":" not in part and "\\" not in part for part in path.parts))


def archive_state(root):
    root = Path(root).resolve()
    output = io.BytesIO()
    total = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for directory in sorted(STATE_DIRS):
            folder = root / directory
            if folder.is_symlink():
                raise ValueError("Links simbólicos não são permitidos no armazenamento.")
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*")):
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    raise ValueError("Links simbólicos não são permitidos no armazenamento.")
                if not path.is_file() or not allowed_member(relative):
                    continue
                total += path.stat().st_size
                if total > MAX_UNPACKED:
                    raise ValueError("A base excede 1 GiB descompactado nesta primeira versão Cloudflare.")
                archive.write(path, relative)
                if output.tell() > MAX_SNAPSHOT:
                    raise ValueError("Snapshot excede 90 MiB nesta primeira versão Cloudflare.")
    content = output.getvalue()
    if len(content) > MAX_SNAPSHOT:
        raise ValueError("Snapshot excede 90 MiB nesta primeira versão Cloudflare.")
    return content


def extract_state(root, content):
    root = Path(root).resolve()
    if len(content) > MAX_SNAPSHOT:
        raise ValueError("Snapshot recebido excede o limite.")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > 100000 or sum(member.file_size for member in members) > MAX_UNPACKED:
            raise ValueError("Snapshot descompactado excede o limite.")
        seen = set()
        for member in members:
            if not allowed_member(member.filename) or member.filename in seen or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Snapshot contém um caminho inválido.")
            seen.add(member.filename)
        # Validate every member before replacing old files, including stale lock files.
        for directory in STATE_DIRS:
            folder = root / directory
            if folder.is_symlink():
                raise ValueError("Diretório de armazenamento não pode ser um link.")
            if folder.exists():
                shutil.rmtree(folder)
        for member in members:
            target = root.joinpath(*PurePosixPath(member.filename).parts)
            if not target.resolve().is_relative_to(root) or target.is_symlink():
                raise ValueError("Snapshot tenta gravar fora da base.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))


class R2State:
    def __init__(self, root, endpoint=None, opener=None):
        self.root = Path(root)
        self.endpoint = (endpoint or os.environ.get("AUTORAG_STORAGE_URL", "http://storage.autorag")).rstrip("/")
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        self.lock = threading.Lock()
        self.last_checkpoint = None

    def request(self, path, content=None):
        headers = {}
        if content is not None:
            headers = {"Content-Type": "application/octet-stream", "X-Content-SHA256": hashlib.sha256(content).hexdigest()}
        request = urllib.request.Request(self.endpoint + path, data=content, method="PUT" if content is not None else "GET", headers=headers)
        try:
            with self.opener(request, timeout=120) as response:
                data = response.read(MAX_SNAPSHOT + 1)
                checksum = response.headers.get("X-Content-SHA256")
                if checksum and hashlib.sha256(data).hexdigest() != checksum:
                    raise OSError("Checksum do armazenamento R2 não confere.")
                return data
        except urllib.error.HTTPError as error:
            if error.code == 404 and content is None:
                return None
            raise OSError("Não foi possível confirmar a persistência no R2.") from error

    def restore(self):
        self.root.mkdir(parents=True, exist_ok=True)
        content = self.request("/snapshot")
        if content is not None:
            extract_state(self.root, content)
        cursor = ""
        while True:
            listing = json.loads(self.request("/jobs" + ("?cursor=" + urllib.parse.quote(cursor, safe="") if cursor else "")))
            for name in listing["keys"]:
                if not name.endswith(".json") or not JOB_ID.fullmatch(name[:-5]):
                    raise ValueError("Identificador de operação inválido no R2.")
                data = self.request("/jobs/" + name)
                if data is None or len(data) > MAX_JOB:
                    raise ValueError("Registro de operação ausente ou muito grande no R2.")
                # Parse before accepting a durable record.
                record = json.loads(data)
                if not isinstance(record, dict) or record.get("job_id") != name[:-5]:
                    raise ValueError("Registro de operação inválido no R2.")
                folder = self.root / "jobs"
                folder.mkdir(exist_ok=True)
                (folder / name).write_bytes(data)
            cursor = listing.get("cursor")
            if not cursor:
                break

    def save_job(self, record):
        job_id = record["job_id"]
        if not JOB_ID.fullmatch(job_id):
            raise ValueError("Identificador de operação inválido.")
        content = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(content) > MAX_JOB:
            raise ValueError("Resultado da operação excede 8 MiB nesta versão Cloudflare.")
        self.request("/jobs/" + job_id + ".json", content)

    def checkpoint(self, project):
        from autorag.core import now
        with self.lock, project.lock():
            self.request("/snapshot", archive_state(self.root))
            self.last_checkpoint = now()
