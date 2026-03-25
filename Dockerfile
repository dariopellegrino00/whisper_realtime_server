# Image with base support for NVIDIA CUDA
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app/

RUN apt-get update && apt-get install -y \
    vim \
    python3 \
    python3-pip \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    tzdata \
    libportaudio2 \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="/root/.cargo/bin:$PATH" && \
    rustup default stable && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

COPY resources /app/resources

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY scripts /app/scripts
COPY src /app/src
COPY Makefile /app/
COPY proto /app/proto
COPY fix_proto_imports.py /app/

RUN python3 /app/scripts/proto.py generate

CMD ["python3", "-u", "-m", "src.server", "--fallback"]
