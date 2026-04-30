# FastAPI service for data-analyst-chatbot.
# Designed for Fly.io / Render / any Docker host. The image runs uvicorn
# against `api.main:app` with the package layout intact.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy the api/ package; the rest of the repo is not needed in the image.
COPY api /app/api

# Vercel-Blob-style /tmp staging dir for runtime data (parquet snapshots).
# DATA_DIR can be overridden if the host gives you a persistent volume.
ENV DATA_DIR=/tmp/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Listen on all interfaces; the host (Fly/Render) maps it to the public port.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
