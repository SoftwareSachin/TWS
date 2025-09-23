FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
WORKDIR /code

# Install system dependencies including build tools, Tesseract, ffmpeg, GStreamer, and Docker
RUN apt clean && apt update && apt install curl -y \
    && apt install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
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
    tesseract-ocr-lat \
    # Add other languages as needed (e.g., tesseract-ocr-fra)
    ffmpeg \
    libavcodec-extra \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*


# Install Node.js (latest LTS version)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

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

# Install project dependencies
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --only main ; fi"

ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install docling==2.32.0

ENV PYTHONPATH=/code
EXPOSE 8000