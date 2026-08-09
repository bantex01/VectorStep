FROM python:3.11-slim AS base

# Non-root runtime user; /data is the single writable volume.
RUN groupadd -g 1000 vectorstep && useradd -m -u 1000 -g vectorstep vectorstep \
    && mkdir -p /data/db /data/artifacts /data/logs /data/pipelines /data/steps \
    && chown -R vectorstep:vectorstep /data

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/src/ ./src/
COPY service/templates/ ./templates/
COPY service/migrations/ ./migrations/
COPY service/alembic.ini ./alembic.ini
COPY samples/ ./samples/

ARG VECTORSTEP_VERSION=dev
ENV VECTORSTEP_VERSION=$VECTORSTEP_VERSION \
    CONFIG_PATH=/etc/vectorstep/config.yaml \
    PYTHONUNBUFFERED=1

USER vectorstep
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
