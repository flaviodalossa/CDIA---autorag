FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AUTORAG_ENV=production AUTORAG_PROJECT_DIR=/data NUMBA_CACHE_DIR=/tmp/numba AUTORAG_STORAGE_URL=http://storage.autorag
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt
COPY autorag ./autorag
COPY embedding_credentials.py api.py ./
COPY web ./web
COPY cloudflare/*.py ./cloudflare/
RUN useradd --create-home --uid 10001 autorag && mkdir /data && chown autorag:autorag /data
USER autorag
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "cloudflare.runtime:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers"]
