# Base image
FROM ubuntu:20.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    cmake \
    ninja-build \
    openjdk-11-jdk \
    texinfo \
    gawk \
    flex \
    bison \
    autoconf \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    device-tree-compiler \
    pkg-config \
    libglib2.0-dev \
    verilator \
    unzip \
    libbsd-dev \
    libjson-c-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --no-cache-dir onnx pyyaml

# Create workspace
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
