# Dockerfile for adversarial ASR attack (M-series Mac compatible)
# Multi-stage build for ARM64 architecture with CPU-only PyTorch

FROM --platform=linux/arm64 ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    build-essential \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies using uv
RUN uv sync --frozen

# Audio data will be mounted at runtime
# Create directory for mounting LibriSpeech data
RUN mkdir -p LibriSpeech

# Make run script executable
RUN chmod +x run_attack.py

# Create output directory
RUN mkdir -p output

# Set default command
CMD ["uv", "run", "python", "run_attack.py"]

# For GPU deployment, uncomment the following lines and use nvidia/cuda base image:
# FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
