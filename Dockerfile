FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Build the search index inside the image so the embedding library version
# used at build time is exactly the one used at runtime.
COPY build_index.py .
COPY data/pages_final.jsonl data/pages_final.jsonl
RUN python build_index.py

COPY app ./app

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
