# syntax=docker/dockerfile:1

# 1. BUILDER

FROM python:3.11.9-slim AS builder

# Some Python packages may need compilers/build tools during installation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 2. RUNTIME

FROM python:3.11.9-slim AS runtime

# Create a non-root user for security.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app

# Copy the already-installed Python environment
# from the builder stage.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY app ./app

RUN mkdir -p /app/uploaded_files \
    && chown -R app:app /app

# Don't run the application as root.
USER app

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status < 500" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]