FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
COPY app.py ./
COPY configs ./configs
COPY migrations ./migrations
COPY references ./references
COPY manifests ./manifests
COPY reports ./reports
RUN useradd --create-home fabguard && mkdir -p runs && chown fabguard runs
USER fabguard
