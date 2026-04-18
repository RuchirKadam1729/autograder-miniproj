FROM python:3.11-slim

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    tesseract-ocr libglib2.0-0 libgl1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install torch/torchvision first from PyTorch's own index
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Then install the rest
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

EXPOSE 7860
CMD ["python", "app.py"]