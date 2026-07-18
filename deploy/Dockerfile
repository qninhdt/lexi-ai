FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY lexi_ai ./lexi_ai
COPY lexi_service ./lexi_service
COPY proto ./proto
COPY alembic.ini ./
COPY alembic ./alembic
COPY deploy ./deploy
RUN pip install uv \
    && uv sync --frozen --extra service --no-dev \
    && /app/.venv/bin/python -m nltk.downloader -d /usr/local/share/nltk_data wordnet

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "--factory", "lexi_service.server:create_app", "--host", "0.0.0.0", "--port", "8443"]
