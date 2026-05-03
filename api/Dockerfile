# FastAPI service for data-analyst-chatbot.
# Built from the api/ directory as the build context (flyctl deploy from api/).
# The api/ contents land at /app/api/ inside the image so the package layout
# `from api.X import Y` works at runtime.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the api/ package contents into /app/api/ — preserves the package
# layout so `from api.X import Y` resolves at runtime.
COPY . /app/api

# Runtime data dir; Fly volume can be mounted here via fly.toml.
ENV DATA_DIR=/tmp/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
