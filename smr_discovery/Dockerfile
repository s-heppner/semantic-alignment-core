FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/

RUN python -m pip install --upgrade pip && pip install .

# Sensible defaults; override at runtime
ENV SMR_ENDPOINTS_JSON_FILE="/app/src/smr_discovery/endpoints.json.default"
ENV SMR_ENDPOINT="http://127.0.0.1"
ENV SMR_LISTEN_ADDRESS="0.0.0.0"
ENV SMR_PORT="8125"

EXPOSE 8125
CMD ["python", "-m", "smr_discovery.service"]
