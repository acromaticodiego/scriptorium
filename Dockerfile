# Scriptorium API image.
#
# The Tesseract *binary* and its language data are system packages, not pip
# packages -- which is the main reason this image exists: it pins the one
# dependency that cannot be installed from PyPI, so the recognition quality in
# CI and in production matches what was measured locally.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# tesseract-ocr-spa / -eng: the language models the OCR engine loads at runtime.
# libglib2.0-0 and libgl1: OpenCV links against them even for headless work.
RUN apt-get update && apt-get install --no-install-recommends -y \
        tesseract-ocr \
        tesseract-ocr-spa \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing source does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,tesseract]"

COPY eval ./eval
COPY web ./web

# Point the app at the client explicitly. The package is installed into
# site-packages while these assets stay in the working directory, so nothing
# should be inferring their location from the module path.
ENV SCR_WEB_DIR=/app/web

# Non-root. The library is the one thing that *does* write, so its directory is
# created and handed over before the switch: /app belongs to root, and a
# bind-mounted host directory does not necessarily arrive writable for uid 10001.
RUN useradd --create-home --uid 10001 scriptorium \
    && mkdir -p /app/data/books /app/out \
    && chown -R scriptorium:scriptorium /app/data /app/out
USER scriptorium

ENV SCR_LIBRARY_DIR=/app/data/books

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "scriptorium.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
