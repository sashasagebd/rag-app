# The web app in one image: the built React UI + FastAPI + the local models.
# Built by the ops-dashboard platform (Docker's legacy builder: no RUN --mount, no heredocs).
#
# Layers are ordered from least to most often changed, so a code-only push rebuilds
# in seconds: the PyTorch, dependency and model layers stay cached.
#
# Runtime env: DATABASE_URL (Postgres with pgvector), APP_SECRET_KEY (required),
# optional ADMIN_USERNAME + ADMIN_PASSWORD_HASH, optional ANTHROPIC_API_KEY.
# Listens on 8000; health check: GET /api/health.

# ---------- 1. build the UI ----------
FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- 2. the app ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models \
    PYTHONPATH=/app/src
WORKDIR /app

# CPU-only PyTorch first (the default wheel bundles ~2 GB of CUDA the server can't use).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The project's dependencies, read from pyproject.toml (the code itself runs from /app/src).
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" \
        > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# The embedding model, its tokenizer and the reranker, named by the app's config,
# downloaded at build time so the container never needs Hugging Face at runtime.
COPY configs/ configs/
COPY src/common/__init__.py src/common/config.py src/common/
RUN python -c "\
from common.config import load_config; \
c = load_config('configs/app.toml'); \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from transformers import AutoTokenizer; \
SentenceTransformer(c['embedding']['model']); \
AutoTokenizer.from_pretrained(c['embedding']['model']); \
CrossEncoder(c['retrieval']['rerank_model']); \
print('models ready:', c['embedding']['model'], c['retrieval']['rerank_model'])"

# The code, the schema and the built UI (served by FastAPI at /).
COPY src/ src/
COPY db/ db/
COPY --from=web /web/dist web/dist

# Run as an unprivileged user; data/ holds only the cache, which may vanish on redeploy.
RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app /app/data
USER app
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    APP_CONFIG=configs/app.toml

EXPOSE 8000
# --proxy-headers: requests arrive through Cloudflare and Caddy; trust their X-Forwarded-* headers.
CMD ["uvicorn", "api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
