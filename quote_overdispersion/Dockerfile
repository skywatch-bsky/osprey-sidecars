# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip uv

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

RUN mkdir -p /app/src/quote_overdispersion && \
    touch /app/src/quote_overdispersion/__init__.py && \
    uv sync --locked --python=$(which python3.11) && \
    pip cache purge && uv cache clean

COPY src/ /app/src/

CMD [".venv/bin/python", "-m", "quote_overdispersion.main"]
