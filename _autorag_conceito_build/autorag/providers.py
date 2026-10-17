from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from embedding_credentials import obter_chave
from .core import digest, split_text

CATALOG = [
    {"provedor": "ollama", "modelo": "bge-m3", "dimensoes": 1024, "limite_tokens": 8192,
     "descricao": "Multilíngue, modelo de referência local", "custo": "local"},
    {"provedor": "ollama", "modelo": "embeddinggemma:300m-qat-q4_0", "dimensoes": 256,
     "dimensoes_nativas": 768, "limite_tokens": 2048, "reducao": "matryoshka",
     "descricao": "Multilíngue leve, Q4, saída reduzida de 256 dimensões", "custo": "local"},
    {"provedor": "sentence_transformers", "modelo": "intfloat/multilingual-e5-small",
     "dimensoes": 384, "limite_tokens": 512, "descricao": "Multilíngue de 384 dimensões nativas", "custo": "local"},
    {"provedor": "sentence_transformers", "modelo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     "dimensoes": 384, "limite_tokens": 128, "descricao": "Multilíngue para campos curtos", "custo": "local"},
    {"provedor": "openai", "modelo": "text-embedding-3-small", "dimensoes": 1536,
     "limite_tokens": 8191, "descricao": "Compatibilidade com índice existente", "custo": "pago"},
    {"provedor": "gemini", "modelo": "gemini-embedding-2", "dimensoes": 768,
     "limite_tokens": 8192, "descricao": "Online, camada gratuita sujeita à conta e cota", "custo": "gratuito_sujeito_cota"},
]


def http_json(url, payload=None, headers=None, timeout=180):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                retry = exc.headers.get("Retry-After", "")
                delay = min(float(retry), 15) if retry.replace(".", "", 1).isdigit() else 2 ** attempt
                time.sleep(delay)
                continue
            detail = exc.read().decode("utf-8", errors="replace")[:700]
            raise RuntimeError(f"Provedor respondeu HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Não foi possível acessar o provedor: {exc.reason}") from exc


def hf_local_path(model, models_dir=None):
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    directories = [models_dir, os.getenv("AUTORAG_MODELS_DIR")]
    for directory in dict.fromkeys(value for value in directories if value):
        candidate = Path(directory) / model.replace("/", "--")
        if candidate.is_dir():
            return candidate.resolve()
    cache = Path(os.getenv("HF_HUB_CACHE") or (Path(os.getenv("HF_HOME", str(Path.home()/".cache"/"huggingface"))) / "hub"))
    repository = cache / ("models--" + model.replace("/", "--"))
    reference = repository / "refs" / "main"
    candidates = [repository/"snapshots"/reference.read_text().strip()] if reference.exists() else []
    if (repository/"snapshots").exists():
        candidates += sorted((repository/"snapshots").iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in candidates:
        if (path/"config.json").exists() and any(path.rglob("*.safetensors")):
            return path.resolve()
    raise ValueError(f"Modelo local não encontrado: {model}. Instale-o com o script 05 ou configure sua pasta.")


def available_models(endpoint="http://127.0.0.1:11434", models_dir=None):
    result = []
    installed = []
    server_error = None
    try:
        installed = http_json(endpoint.rstrip("/") + "/api/tags", timeout=4).get("models", [])
    except Exception as exc:
        server_error = str(exc)
    for entry in CATALOG:
        row = dict(entry)
        if row["provedor"] == "ollama":
            names = {item.get("name", "") for item in installed}
            row["disponivel"] = row["modelo"] in names or row["modelo"] + ":latest" in names
            row["detalhe"] = "Instalado" if row["disponivel"] else (server_error or "Modelo não instalado")
        elif row["provedor"] == "sentence_transformers":
            try:
                row["caminho_local"] = str(hf_local_path(row["modelo"], models_dir))
                row["disponivel"] = importlib.util.find_spec("sentence_transformers") is not None
                row["detalhe"] = "Arquivos locais encontrados" if row["disponivel"] else "Falta instalar dependência local"
            except ValueError:
                row["disponivel"] = False
                row["detalhe"] = "Modelo não instalado no cache/pasta configurada"
        else:
            row["disponivel"] = bool(obter_chave(row["provedor"]))
            row["detalhe"] = "Chave configurada; disponibilidade depende da conta" if row["disponivel"] else "Configure a chave"
        result.append(row)
    known = {row["modelo"] for row in CATALOG if row["provedor"] == "ollama"}
    for model in installed:
        name = model.get("name", "")
        if name in known or name.removesuffix(":latest") in known:
            continue
        try:
            description = http_json(endpoint.rstrip("/") + "/api/show", {"model": name}, timeout=6)
            if "embedding" not in description.get("capabilities", []):
                continue
            info = description.get("model_info", {})
            dims = next((int(value) for key, value in info.items() if key.endswith(".embedding_length")), None)
            limit = next((int(value) for key, value in info.items() if key.endswith(".context_length")), 512)
            result.append({"provedor": "ollama", "modelo": name, "dimensoes": dims,
                           "limite_tokens": limit, "disponivel": True, "custo": "local",
                           "descricao": "Modelo de embeddings instalado", "detalhe": "Capacidade verificada"})
        except Exception:
            continue
    return result


def profile_spec(provider, model, dimensions=None, endpoint=None, models_dir=None):
    known = next((dict(row) for row in CATALOG if row["provedor"] == provider and row["modelo"] == model), {})
    known.pop("descricao", None)
    known.pop("custo", None)
    known.update({"provedor": provider, "modelo": model, "dimensoes": dimensions or known.get("dimensoes"),
                  "endpoint": (endpoint or ("http://127.0.0.1:11434" if provider == "ollama" else "")).rstrip("/"),
                  "normalizacao": "l2", "versao_preparacao": "1", "segmentacao": "utf8_conservador_v1"})
    if provider == "ollama":
        tags = http_json(known["endpoint"] + "/api/tags", timeout=8).get("models", [])
        found = next((item for item in tags if item.get("name") in {model, model+":latest"}), None)
        if not found:
            raise ValueError(f"O modelo {model} não está instalado no Ollama.")
        description = http_json(known["endpoint"] + "/api/show", {"model": model}, timeout=30)
        capabilities = description.get("capabilities")
        if capabilities and "embedding" not in capabilities:
            raise ValueError("O modelo selecionado não oferece embeddings.")
        info = description.get("model_info", {})
        native = next((int(value) for key, value in info.items() if key.endswith(".embedding_length")), known.get("dimensoes_nativas") or known.get("dimensoes"))
        known["dimensoes_nativas"] = native
        known["dimensoes"] = dimensions or known.get("dimensoes") or native
        known["limite_tokens"] = min(known.get("limite_tokens", 8192), next((int(value) for key, value in info.items() if key.endswith(".context_length")), 8192))
        known["revisao_modelo"] = found["digest"]
        if known["dimensoes"] != native and not model.startswith("embeddinggemma"):
            raise ValueError("Redução local de dimensão é habilitada somente para EmbeddingGemma (Matryoshka).")
        if model.startswith("embeddinggemma"):
            known["reducao"] = "matryoshka" if known["dimensoes"] != native else "nenhuma"
            if known["dimensoes"] not in {128, 256, 512, 768}:
                raise ValueError("EmbeddingGemma: escolha 128, 256, 512 ou 768 dimensões.")
    elif provider == "sentence_transformers":
        path = hf_local_path(model, models_dir)
        known["caminho_local"] = str(path)
        # Hash dos arquivos locais, não somente nome da pasta: distingue mudanças de pesos.
        from .core import file_hash
        files = sorted([*path.rglob("*.safetensors"), *path.rglob("*.bin"), *path.rglob("*.json"),
                        *path.rglob("*.model"), *path.rglob("*.txt")])
        known["revisao_modelo"] = digest([(str(p.relative_to(path)), file_hash(p)) for p in files])
        backend = SentenceProvider(known)
        known["dimensoes"] = backend.model.get_sentence_embedding_dimension()
        if dimensions is not None and dimensions != known["dimensoes"]:
            raise ValueError("Use a dimensão nativa desse modelo local.")
        known["limite_tokens"] = int(backend.model.max_seq_length)
    elif provider in {"openai", "gemini"}:
        if not obter_chave(provider):
            raise ValueError(f"Configure a chave de {provider} em embedding_credentials.py ou variável de ambiente.")
        known["revisao_modelo"] = model  # Serviços sem digest: nome público registrado.
        known.setdefault("limite_tokens", 8191 if provider == "openai" else 8192)
    else:
        raise ValueError(f"Provedor desconhecido: {provider}")
    if not isinstance(known["dimensoes"], int) or known["dimensoes"] <= 0:
        raise ValueError("Informe a dimensão positiva do modelo.")
    known["perfil_id"] = "pf_" + digest(known)[:20]
    return known


class Provider:
    def __init__(self, profile):
        self.profile = profile

    def prepare(self, value, role):
        model = self.profile["modelo"].lower()
        if "multilingual-e5" in model:
            return ("query: " if role == "consulta" else "passage: ") + value
        if "embeddinggemma" in model or model == "gemini-embedding-2":
            return ("task: search result | query: " if role == "consulta" else "title: none | text: ") + value
        return value

    def segments(self, value, role="documento"):
        overhead = len(self.prepare("", role).encode("utf-8"))
        budget = self.profile.get("limite_tokens", 512) - overhead - 24
        return list(split_text(value, budget))

    def embed(self, texts, role="documento"):
        raise NotImplementedError


class OllamaProvider(Provider):
    def __init__(self, profile):
        super().__init__(profile)
        tags = http_json(profile["endpoint"] + "/api/tags", timeout=8).get("models", [])
        found = next((row for row in tags if row.get("name") in {profile["modelo"], profile["modelo"]+":latest"}), None)
        if not found or found.get("digest") != profile["revisao_modelo"]:
            raise ValueError("Modelo Ollama ausente ou versão alterada. Cadastre um novo perfil para o digest atual.")

    def embed(self, texts, role="documento"):
        raw_batch = os.getenv("AUTORAG_OLLAMA_INPUT_BATCH", "0")
        if not raw_batch.isdigit() or int(raw_batch) > 8192:
            raise ValueError("AUTORAG_OLLAMA_INPUT_BATCH deve ser um inteiro entre 0 e 8192.")
        batch_size = int(raw_batch) or max(1, len(texts))
        keep_alive = os.getenv("AUTORAG_OLLAMA_KEEP_ALIVE", "5m")
        keep_alive = 0 if keep_alive == "0" else keep_alive
        raw_threads = os.getenv("AUTORAG_OLLAMA_NUM_THREAD", "")
        if raw_threads and (not raw_threads.isdigit() or not 1 <= int(raw_threads) <= 256):
            raise ValueError("AUTORAG_OLLAMA_NUM_THREAD deve ser um inteiro entre 1 e 256.")
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            # Keep weights loaded between small CPU batches, then release before UMAP.
            final = start + batch_size >= len(texts)
            payload = {"model": self.profile["modelo"],
                       "input": [self.prepare(t, role) for t in batch], "truncate": False,
                       "keep_alive": keep_alive if final else "5m"}
            if raw_threads:
                payload["options"] = {"num_thread": int(raw_threads)}
            result = http_json(self.profile["endpoint"] + "/api/embed", payload)
            values = np.asarray(result.get("embeddings"), dtype=np.float64)
            # Preserve the existing explicit Matryoshka reduction and input ordering.
            chunks.append(self._dimension(values, len(batch)))
        return np.concatenate(chunks) if chunks else np.empty((0, self.profile["dimensoes"]))

    def _dimension(self, values, count):
        expected = self.profile["dimensoes"]
        if values.ndim != 2 or len(values) != count:
            raise ValueError("O provedor retornou uma quantidade inesperada de embeddings.")
        if values.shape[1] != expected:
            if self.profile.get("reducao") != "matryoshka" or values.shape[1] < expected:
                raise ValueError(f"Dimensão recebida {values.shape[1]}, esperada {expected}.")
            values = values[:, :expected].copy()
        return values


class SentenceProvider(Provider):
    def __init__(self, profile):
        super().__init__(profile)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Instale os opcionais locais pelo script 05_instalar_modelo_leve.bat.") from exc
        options = {"local_files_only": True, "trust_remote_code": False}
        device = os.getenv("AUTORAG_SENTENCE_DEVICE", "").strip()
        if device:
            options["device"] = device
        threads = os.getenv("AUTORAG_SENTENCE_NUM_THREADS", "")
        if threads:
            if not threads.isdigit() or not 1 <= int(threads) <= 256:
                raise ValueError("AUTORAG_SENTENCE_NUM_THREADS deve ser inteiro entre 1 e 256.")
            import torch
            torch.set_num_threads(int(threads))
        self.model = SentenceTransformer(profile["caminho_local"], **options)

    def segments(self, value, role="documento"):
        # Conservador por bytes; confere com o tokenizador real antes de enviar.
        pieces = super().segments(value, role)
        for segment, _, _ in pieces:
            count = len(self.model.tokenizer.encode(self.prepare(segment, role), add_special_tokens=True))
            if count > self.model.max_seq_length:
                raise ValueError("Segmento excede o limite real do modelo local.")
        return pieces

    def embed(self, texts, role="documento"):
        options = {"normalize_embeddings": False, "convert_to_numpy": True}
        batch_size = os.getenv("AUTORAG_SENTENCE_BATCH_SIZE", "")
        if batch_size:
            if not batch_size.isdigit() or not 1 <= int(batch_size) <= 8192:
                raise ValueError("AUTORAG_SENTENCE_BATCH_SIZE deve ser inteiro entre 1 e 8192.")
            options.update(batch_size=int(batch_size), show_progress_bar=False)
        prepared = [self.prepare(value, role) for value in texts]
        for value in prepared:
            if len(self.model.tokenizer.encode(value, add_special_tokens=True)) > self.model.max_seq_length:
                raise ValueError("Entrada excede o limite do modelo; nenhum truncamento automático será aceito.")
        return np.asarray(self.model.encode(prepared, **options))


class OpenAIProvider(Provider):
    def embed(self, texts, role="documento"):
        key = obter_chave("openai")
        if not key:
            raise ValueError("Chave OpenAI não configurada.")
        request = {"model": self.profile["modelo"], "input": [self.prepare(value, role) for value in texts], "encoding_format": "float"}
        if self.profile["modelo"].startswith("text-embedding-3"):
            request["dimensions"] = self.profile["dimensoes"]
        response = http_json("https://api.openai.com/v1/embeddings", request, {"Authorization": "Bearer " + key})
        data = sorted(response["data"], key=lambda item: item["index"])
        return np.asarray([item["embedding"] for item in data], dtype=np.float64)


class GeminiProvider(Provider):
    def embed(self, texts, role="documento"):
        key = obter_chave("gemini")
        if not key:
            raise ValueError("Chave Gemini não configurada.")
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + urllib.parse.quote(self.profile["modelo"], safe="") + ":embedContent"
        values = []
        # Embedding 2 agrega múltiplos textos: uma chamada padrão por unidade.
        for value in texts:
            payload = {"model": "models/" + self.profile["modelo"],
                       "content": {"parts": [{"text": self.prepare(value, role)}]},
                       "outputDimensionality": self.profile["dimensoes"]}
            if self.profile["modelo"] == "gemini-embedding-001":
                payload["taskType"] = "RETRIEVAL_QUERY" if role == "consulta" else "RETRIEVAL_DOCUMENT"
            response = http_json(url, payload, {"x-goog-api-key": key})
            values.append(response["embedding"]["values"])
        return np.asarray(values, dtype=np.float64)


def get_provider(profile):
    classes = {"ollama": OllamaProvider, "sentence_transformers": SentenceProvider,
               "openai": OpenAIProvider, "gemini": GeminiProvider}
    if profile["provedor"] not in classes:
        raise ValueError("Provedor não suportado.")
    return classes[profile["provedor"]](profile)
