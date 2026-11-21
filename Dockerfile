FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY index.pkl /app/index.pkl
ENV PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/index.pkl HOST=0.0.0.0 PORT=8000 PYTHONPATH=/app/src
USER appuser
EXPOSE 8000
CMD ["findex", "serve", "--port", "8000"]
