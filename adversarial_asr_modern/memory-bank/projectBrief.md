# Project Brief: Adversarial ASR Modern

## Objective
Modernize the 2017 Python 2 adversarial audio attack system to work with Python 3, modern libraries, and containerized deployment.

## Core Requirements
- Convert Python 2 codebase to Python 3
- Replace TensorFlow 1.x with modern PyTorch
- Replace lingvo ASR model with OpenAI Whisper
- Create containerized solution that runs on M-series Mac (CPU) and can be deployed to cloud GPU
- Process 10 audio samples as initial milestone

## Original System Overview
The original system implements a two-stage adversarial attack:
1. **Stage 1**: Optimize adversarial perturbations to fool the ASR model
2. **Stage 2**: Refine perturbations to be imperceptible using psychoacoustic masking

Key components:
- `generate_imperceptible_adv.py`: Main attack script
- `generate_masking_threshold.py`: Psychoacoustic masking calculations
- `tool.py`: Utility functions for audio processing and model interfacing
- Audio samples from LibriSpeech dataset

## Technology Migration
- **From**: Python 2.7, TensorFlow 1.14, lingvo ASR model
- **To**: Python 3.9+, PyTorch 2.x, OpenAI Whisper via transformers

## Success Criteria
- Docker container successfully builds and runs
- Processes 10 LibriSpeech audio samples
- Generates adversarial audio files (.wav outputs)
- Attack effectiveness validated against Whisper model
