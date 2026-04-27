# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_INDEX_STRATEGY=unsafe-best-match \
    HF_HOME=/opt/hf-cache

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

RUN uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/privacy-filter')"


FROM python:3.12-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @llamaindex/liteparse \
 && npm cache clean --force \
 && apt-get purge -y --auto-remove curl gnupg \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1001 hush \
 && useradd --system --uid 1001 --gid 1001 --home /app --shell /usr/sbin/nologin hush

WORKDIR /app

COPY --from=builder --chown=hush:hush /app/.venv /app/.venv
COPY --from=builder --chown=hush:hush /opt/hf-cache /opt/hf-cache
COPY --chown=hush:hush app.py ./
COPY --chown=hush:hush assets/ ./assets/
COPY --chown=hush:hush probe/ ./probe/

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HUSH_SESSION_TTL_SECONDS=900 \
    npm_config_cache=/tmp/.npm \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false \
    NO_UPDATE_NOTIFIER=true

USER hush

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=3); sys.exit(0)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765", "--proxy-headers", "--forwarded-allow-ips", "*"]
