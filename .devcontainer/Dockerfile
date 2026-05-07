FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake ninja-build \
    rsync openssh-client sshpass tmux \
    gdb-multiarch \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    git curl wget unzip ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace