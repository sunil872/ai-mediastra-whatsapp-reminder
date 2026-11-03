# RefillCare - one image, three roles (pick with the command):
#   API (default) : uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
#   worker        : python -m refillcare.jobs.worker
#   streamlit     : streamlit run app_refillcare.py --server.address 0.0.0.0 --server.port 8501
#   migrations    : python -m database.migrations --all-stores      (run once per deploy)
#
# Configuration comes only from environment variables (ECS task definition / Secrets Manager):
# DATABASE_URL, REFILLCARE_API_KEY, WHATSAPP_WEBHOOK_SECRET, XINNO_*, ...
# No .env and no patient data are baked into the image (.dockerignore).

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1 \
    REFILLCARE_ENV=production \
    REFILLCARE_LOG_FORMAT=json \
    REFILLCARE_DATA_DIR=/data/refillcare

WORKDIR /app

# Wheels exist for every pinned package on linux/amd64 + arm64; libgomp is needed by scikit-learn / xgboost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Non-root user; /data holds the per-store working folder (models cache, catalogue) - mount a volume
RUN useradd --create-home --uid 10001 refillcare \
    && mkdir -p /data/refillcare/processed \
    && chown -R refillcare:refillcare /app /data
USER refillcare

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=4).status == 200 else 1)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
