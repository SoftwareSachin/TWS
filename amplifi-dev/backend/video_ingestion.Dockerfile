# Build stage
FROM python:3.12-slim as builder

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
WORKDIR /code

ARG ENABLE_GPU=true
ARG CUDA_VERSION=12.6
ARG ENABLE_VIDEO_INGESTION=true

# Install system dependencies conditionally based on video ingestion
RUN apt clean && apt update && apt install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    gcc \
    g++ \
    && if [ "$ENABLE_VIDEO_INGESTION" = "true" ] ; then \
    echo "Installing video processing system dependencies..." && \
    apt install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 ; \
    else \
    echo "Video ingestion disabled - skipping video system dependencies" ; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.3 POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

ENV POETRY_HTTP_TIMEOUT=300

# Copy poetry.lock* in case it doesn't exist in the repo
COPY app/pyproject.toml app/poetry.lock* /code/

# Regenerate lock file to match pyproject.toml
RUN poetry lock --no-update

# Install project dependencies with cleanup (including dev dependencies for testing)
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root --no-cache ; else poetry install --no-root --only main --no-cache ; fi" \
    && rm -rf ~/.cache/pip \
    && rm -rf ~/.cache/pypoetry \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy video requirements file (always copy, but conditionally install)
COPY app/app/utils/video_ingestion/requirements.txt /tmp/video_requirements.txt

# Install video processing dependencies with cleanup (only if video ingestion is enabled)
RUN if [ "$ENABLE_VIDEO_INGESTION" = "true" ] ; then \
    echo "Video ingestion enabled - installing video processing dependencies..." && \
    if [ "$ENABLE_GPU" = "true" ] ; then \
    echo "Installing GPU-enabled video processing dependencies..." && \
    export CUDA_VERSION=12.6 && \
    export FORCE_CUDA=1 && \
    export CUDA_HOME=/usr/local/cuda && \
    # Check if we're in a CI environment or if GPU is actually available
    if nvidia-smi >/dev/null 2>&1; then \
    echo "GPU detected - installing with full GPU support" && \
    pip install --no-cache-dir -r /tmp/video_requirements.txt ; \
    else \
    echo "No GPU detected (likely CI environment) - installing with CPU fallback for bitsandbytes" && \
    # Install most packages normally, but handle bitsandbytes specially
    grep -v "bitsandbytes" /tmp/video_requirements.txt | pip install --no-cache-dir -r /dev/stdin && \
    # Install CPU version of bitsandbytes for CI builds
    pip install --no-cache-dir bitsandbytes --prefer-binary --no-deps ; \
    fi && \
    echo "Configuring cuDNN library paths for production..." && \
    echo "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib" > /etc/ld.so.conf.d/nvidia-cudnn.conf && \
    echo "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib" > /etc/ld.so.conf.d/nvidia-cublas.conf && \
    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/nvidia-cuda.conf && \
    ldconfig && \
    echo "Library paths configured permanently" ; \
    else \
    echo "Installing CPU-only video processing dependencies..." && \
    sed 's/+cu126//g' /tmp/video_requirements.txt | \
    sed 's|--extra-index-url https://download.pytorch.org/whl/cu126|--extra-index-url https://download.pytorch.org/whl/cpu|g' | \
    pip install --no-cache-dir -r /dev/stdin ; \
    fi && \
    rm -rf ~/.cache/pip && \
    rm -rf /tmp/video_requirements.txt ; \
    else \
    echo "Video ingestion disabled - skipping video processing dependencies" && \
    echo "Creating empty requirements file to avoid build errors" && \
    echo "# Video ingestion disabled" > /tmp/video_requirements.txt ; \
    fi

# Copy application code
COPY app/alembic.ini /code/
COPY app/alembic /code/alembic
COPY app/app /code/app
COPY app/run_celery_video.py /code/

# Pre-download video processing models (only if video ingestion and GPU are enabled)
RUN if [ "$ENABLE_VIDEO_INGESTION" = "true" ] && [ "$ENABLE_GPU" = "true" ] ; then \
    echo "Checking environment for model downloads..." && \
    mkdir -p ~/.cache/huggingface && \
    export HF_HUB_DISABLE_PROGRESS_BARS=1 && \
    export HF_HUB_DISABLE_TELEMETRY=1 && \
    export CUDA_VISIBLE_DEVICES="" && \
    export BITSANDBYTES_NOWELCOME=1 && \
    # Check if GPU is available for model downloads
    if nvidia-smi >/dev/null 2>&1; then \
    echo "GPU detected - proceeding with model downloads..." && \
    python -c "exec('''try:\n    import os\n    os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"\"\n    os.environ[\"BITSANDBYTES_NOWELCOME\"] = \"1\"\n    \n    print(\"=== Downloading Video Processing Models ===\")\n    print(\"Using standard cache locations (~/.cache/huggingface)\")\n    \n    # Download MiniCPM-V Caption Model (download only, no loading)\n    print(\"1. Downloading MiniCPM-V-2_6-int4 caption model...\")\n    from huggingface_hub import snapshot_download\n    caption_model_name = \"openbmb/MiniCPM-V-2_6-int4\"\n    revision = \"06219bd\"\n    \n    # Just download the model files without loading - use absolute cache path\n    cache_dir = os.path.expanduser(\"~/.cache/huggingface\")\n    snapshot_download(\n        repo_id=caption_model_name,\n        revision=revision,\n        cache_dir=cache_dir\n    )\n    print(\"   ✓ Caption model downloaded successfully\")\n    \n    # Download Whisper Model using huggingface_hub to avoid faster-whisper dependencies\n    print(\"2. Downloading Whisper transcription model...\")\n    whisper_model_name = \"Systran/faster-distil-whisper-large-v3\"\n    \n    # Download whisper model using huggingface_hub to avoid ctranslate2 issues\n    snapshot_download(\n        repo_id=whisper_model_name,\n        cache_dir=cache_dir\n    )\n    print(\"   ✓ Whisper model downloaded successfully\")\n    \n    print(\"=== All video models downloaded to standard cache ===\")\n    \nexcept Exception as e:\n    print(f\"Model download failed: {e}\")\n    print(\"Models will be downloaded at runtime instead\")\n''')" ; \
    else \
    echo "No GPU detected (likely CI environment) - skipping model downloads" && \
    echo "Models will be downloaded at runtime when GPU is available" ; \
    fi ; \
    elif [ "$ENABLE_VIDEO_INGESTION" = "false" ] ; then \
    echo "Video ingestion disabled - skipping model downloads" ; \
    else \
    echo "GPU disabled - skipping model pre-download (will download at runtime if needed)" ; \
    fi

# GPU environment configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Production-ready library path configuration (fallback for ldconfig)
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Hugging Face optimization settings
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false
# Python path
ENV PYTHONPATH=/code

EXPOSE 8000
