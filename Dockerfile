# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System packages
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
        libgl1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# Hugging Face Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Cache dir for transformers / torch hub
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

EXPOSE 7860

CMD ["python", "app.py"]
