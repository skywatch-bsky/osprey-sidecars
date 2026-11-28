# syntax=docker/dockerfile:1
FROM python:3.14-slim

WORKDIR /app

RUN pip install --upgrade pip uv

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

RUN mkdir -p /app/src/faction_analysis && \
    touch /app/src/faction_analysis/__init__.py && \
    uv sync --locked --no-group dev --python=$(which python3.14) && \
    pip cache purge && uv cache clean

COPY sql/ /app/sql/
COPY src/ /app/src/

CMD [".venv/bin/python", "-m", "faction_analysis.main"]
