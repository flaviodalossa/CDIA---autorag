FROM python:3.11-slim AS model-assets
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY cloudflare/__init__.py cloudflare/download_models.py cloudflare/install_ollama.py cloudflare/models.lock.json ./cloudflare/
RUN python -m cloudflare.install_ollama --root /opt/ollama-cpu
RUN python -m cloudflare.download_models --root /opt/models/ollama

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AUTORAG_ENV=production AUTORAG_PROJECT_DIR=/data NUMBA_CACHE_DIR=/tmp/numba AUTORAG_STORAGE_URL=http://storage.autorag \
    OLLAMA_MODELS=/opt/models/ollama OLLAMA_HOST=127.0.0.1:11434 OLLAMA_BASE_URL=http://127.0.0.1:11434 OLLAMA_NO_CLOUD=1 \
    OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_QUEUE=8 OLLAMA_KEEP_ALIVE=0 OLLAMA_VULKAN=0 \
    AUTORAG_OLLAMA_KEEP_ALIVE=0 AUTORAG_OLLAMA_INPUT_BATCH=1 AUTORAG_OLLAMA_NUM_THREAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 libopenblas0-pthread && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt
COPY autorag ./autorag
COPY embedding_credentials.py api.py ./
COPY web ./web
COPY cloudflare/*.py ./cloudflare/
COPY cloudflare/models.lock.json ./cloudflare/
COPY --from=model-assets /opt/ollama-cpu/bin/ollama /usr/bin/ollama
COPY --from=model-assets /opt/ollama-cpu/lib/ollama /usr/lib/ollama
# Ollama migrates registry metadata into manifests-v2 at runtime. Set ownership
# while copying the assets so this does not duplicate the 1.4 GB weight layer.
COPY --from=model-assets --chown=10001:10001 /opt/models/ollama /opt/models/ollama
RUN useradd --create-home --uid 10001 autorag && mkdir /data && chown autorag:autorag /data
USER autorag
EXPOSE 8000
CMD ["python", "-m", "cloudflare.boot"]
