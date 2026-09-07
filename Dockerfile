# --- build: resolve dependências com uv em uma layer cacheável ---
FROM python:3.12-slim AS build
WORKDIR /app

# uv (gerenciador de dependências)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Instala só as dependências primeiro (melhor cache); depois o código.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# README.md é referenciado por [project].readme no pyproject.toml, então o
# build backend (hatchling) precisa dele ao instalar o próprio projeto.
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# --- runtime: imagem enxuta, sem uv, usuário não-root ---
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN groupadd --system app && useradd --system --gid app app

COPY --from=build /opt/venv /opt/venv
COPY src ./src

ENV PATH="/opt/venv/bin:$PATH"

USER app:app

EXPOSE 8000
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
