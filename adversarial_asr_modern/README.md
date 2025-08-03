# Adversarial ASR Modern

A modernized implementation of adversarial audio attacks against Automatic Speech Recognition (ASR) systems, ported from the original 2017 Python 2/TensorFlow 1.x implementation to Python 3/PyTorch with OpenAI Whisper.

## Overview

This project implements a two-stage adversarial attack:
1. **Stage 1**: Optimize adversarial perturbations to fool the ASR model
2. **Stage 2**: Refine perturbations to be imperceptible using psychoacoustic masking

The modernized version replaces:
- Python 2.7 → Python 3.9+
- TensorFlow 1.x → PyTorch 2.x
- Lingvo ASR → OpenAI Whisper
- Manual dependency management → UV package manager

## Quick Start with Docker

### Prerequisites
- Docker installed on your system
- The original `adversarial_asr` directory (contains LibriSpeech audio samples)

### Audio Samples Source
The attack uses 10 audio samples from the **LibriSpeech test-clean dataset** that are already included in the original `adversarial_asr/LibriSpeech/test-clean/` directory. These are the exact same samples used in the original 2017 research.

### Build and Run

1. **Build the Docker image:**
   ```bash
   cd adversarial_asr_modern
   docker build -t adversarial-asr-modern .
   ```

2. **Run the container with audio files mounted:**
   ```bash
   # Mount the LibriSpeech directory from the original project
   docker run -v "$(pwd)/../adversarial_asr/LibriSpeech:/app/LibriSpeech" \
              -v "$(pwd)/output:/app/output" \
              adversarial-asr-modern
   ```

3. **Check results:**
   The adversarial audio files will be saved in the `output/` directory.

### Alternative: Run without Docker

If you prefer to run locally:

1. **Install UV:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Install dependencies:**
   ```bash
   cd adversarial_asr_modern
   uv sync
   ```

3. **Run the attack:**
   ```bash
   uv run python run_attack.py
   ```

## Project Structure

```
adversarial_asr_modern/
├── src/adversarial_asr_modern/     # Main package
│   ├── __init__.py
│   ├── adversarial_attack.py       # Main attack implementation
│   ├── audio_utils.py              # Whisper model and audio utilities
│   └── masking_threshold.py        # Psychoacoustic masking functions
├── memory-bank/                    # Documentation and progress tracking
├── run_attack.py                   # Simple test runner
├── Dockerfile                      # Container definition
├── pyproject.toml                  # Dependencies and project config
└── README.md                       # This file
```

## Features

- **Device Agnostic**: Automatically detects and uses GPU when available, falls back to CPU
- **ARM64 Compatible**: Designed to run on M-series Macs via Docker
- **Modern Dependencies**: Uses latest PyTorch, Transformers, and audio processing libraries
- **Containerized**: Easy deployment and reproducible results
- **Modular Design**: Clean separation between audio processing, masking, and attack logic

## Configuration

The attack can be configured via command-line arguments or by modifying the parameters in `run_attack.py`:

- `model_name`: Whisper model to use (default: "openai/whisper-base")
- `device`: Computation device ("cpu" or "cuda")
- `batch_size`: Number of audio samples to process simultaneously
- `num_iter_stage1/stage2`: Number of optimization iterations per stage

## GPU Deployment

For cloud GPU deployment, modify the Dockerfile base image:

```dockerfile
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
```

And ensure NVIDIA Docker runtime is available.

## Original Research

This implementation is based on the paper:
"Audio Adversarial Examples: Targeted Attacks on Speech-to-Text"
by Nicholas Carlini and David Wagner (2018)

## License

This project maintains compatibility with the original cleverhans library license.
