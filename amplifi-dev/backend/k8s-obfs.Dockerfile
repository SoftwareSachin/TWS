# Stage 1: Build
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy pyproject.toml and poetry.lock
COPY app/pyproject.toml app/poetry.lock* /code/

# Install dependencies with Poetry
WORKDIR /code
RUN poetry install --no-root --only main

# Install PyArmor
RUN pip install pyarmor==9.1.7

# Copy the PyArmor CI license file
COPY py-files/pyarmor-ci.zip /code/

# Register PyArmor with CI license and debug
RUN poetry run pyarmor reg /code/pyarmor-ci.zip || (cat ~/.pyarmor/pyarmor.error.log && exit 1) && \
    ls -la /root/.pyarmor || (echo ".pyarmor not found in /root" && exit 1)

# Copy the application code, entry point scripts, and Alembic migrations
COPY app/app /code/app
COPY app/run_fastapi.py app/run_celery.py app/run_celery_beat.py /code/
COPY app/alembic.ini /code/
COPY app/alembic /code/alembic

# Recursively obfuscate all Python files under app/ with PyArmor
RUN poetry run pyarmor gen --recursive app/ || (cat ~/.pyarmor/pyarmor.error.log && exit 1) && \
    poetry run pyarmor gen run_fastapi.py || (cat ~/.pyarmor/pyarmor.error.log && exit 1) && \
    poetry run pyarmor gen run_celery.py || (cat ~/.pyarmor/pyarmor.error.log && exit 1) && \
    poetry run pyarmor gen run_celery_beat.py || (cat ~/.pyarmor/pyarmor.error.log && exit 1) && \
    find /code/dist/app -type d -exec touch {}/__init__.py \; && \
    ls -la /code/dist || (echo "dist directory not found" && exit 1) && \
    ls -la /code/dist/app || (echo "dist/app directory not found" && exit 1)

# Stage 2: Runtime
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PYTHONPATH=/app:/app/dist

# Install system dependencies, including curl, Tesseract, ffmpeg, and supervisor
RUN apt clean && apt update && apt install curl -y \
    && apt install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-rus \
    tesseract-ocr-ara \
    tesseract-ocr-hin \
    tesseract-ocr-nld \
    tesseract-ocr-tur \
    ffmpeg \
    libavcodec-extra \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (latest LTS version)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
&& apt-get install -y nodejs

# Install Poetry for runtime dependency installation
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy pyproject.toml and poetry.lock for runtime dependencies
COPY app/pyproject.toml app/poetry.lock* /app/

# Install runtime dependencies with Poetry and debug
WORKDIR /app
RUN poetry install --no-root --only main && \
    poetry run pip list

# Install PyArmor runtime dependencies
RUN pip install pyarmor==9.1.7

ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1
ENV OMP_NUM_THREADS=1
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install docling==2.32.0

# Copy PyArmor runtime library from builder stage
COPY --from=builder /root/.pyarmor /root/.pyarmor

# Copy the obfuscated files from the builder stage
COPY --from=builder /code/dist /app/dist

# Copy Alembic migration files
COPY --from=builder /code/alembic.ini /app/
COPY --from=builder /code/alembic /app/alembic

# Copy supervisord configuration files
COPY supervisord-fastapi.conf /etc/supervisor/conf.d/supervisord-fastapi.conf
COPY supervisord-celery-default.conf /etc/supervisor/conf.d/supervisord-celery-default.conf
COPY supervisord-celery-ingestion.conf /etc/supervisor/conf.d/supervisord-celery-ingestion.conf
COPY supervisord-celery-file-compression.conf /etc/supervisor/conf.d/supervisord-celery-file-compression.conf
COPY supervisord-celery-beat.conf /etc/supervisor/conf.d/supervisord-celery-beat.conf

# Debug runtime environment
RUN ls -la /app/dist || echo "dist not found in runtime" && \
    ls -la /app/dist/app || echo "dist/app not found in runtime"

# Expose port for FastAPI
EXPOSE 8000

# Default command (overridden by deployment YAMLs)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord-fastapi.conf"]