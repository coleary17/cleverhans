# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains adversarial audio attack research implementations targeting Automatic Speech Recognition (ASR) systems. It consists of two main components:

1. **adversarial_asr/**: Original 2017 implementation (Python 2.7, TensorFlow 1.x, Lingvo ASR)
2. **adversarial_asr_modern/**: Modernized implementation (Python 3.9+, PyTorch 2.x, OpenAI Whisper)

Both implementations demonstrate targeted adversarial attacks that modify audio to make ASR systems produce desired transcriptions while keeping modifications imperceptible.

## Development Commands

### Modern Implementation (adversarial_asr_modern)

#### Environment Setup
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
cd adversarial_asr_modern
uv sync
```

#### Running Attacks
```bash
# Standard Whisper attack
uv run python run_attack.py

# CTC model attack (wav2vec2)
uv run python run_ctc_attack.py
```

#### Testing
```bash
# Test installation
uv run python test_installation.py

# Test audio outputs
uv run python test_audio_outputs.py

# Test CTC model
uv run python test_ctc_model.py

# Debug Whisper model
uv run python debug_whisper.py

# Diagnose audio distortion
uv run python diagnose_distortion.py
```

#### Docker Operations
```bash
# Build Docker image
docker build -t adversarial-asr-modern .

# Run with mounted audio files
docker run -v "$(pwd)/../adversarial_asr/LibriSpeech:/app/LibriSpeech" \
           -v "$(pwd)/output:/app/output" \
           adversarial-asr-modern
```

### Original Implementation (adversarial_asr)

#### Running Attacks
```bash
cd adversarial_asr

# Generate imperceptible adversarial examples
python generate_imperceptible_adv.py

# Generate robust adversarial examples (over-the-air simulation)
python room_simulator.py  # First generate room reverberations
python generate_robust_adv.py --initial_bound=2000 --num_iter_stage1=2000
```

#### Testing
```bash
# Test imperceptible adversarial examples
python test_imperceptible_adv.py --stage=stage2 --adv=True

# Test robust adversarial examples
python test_robust_adv.py --stage=stage2 --adv=True

# Test basic components
python test_basic_components.py

# Test full system
python test_full_system.py
```

## Architecture Overview

### Modern Implementation Structure

The modernized implementation follows a modular architecture with clear separation of concerns:

**Core Attack Pipeline:**
1. `audio_utils.py` / `ctc_audio_utils.py`: Handles model loading (Whisper/wav2vec2) and audio processing
2. `adversarial_attack.py` / `ctc_attack.py`: Implements the two-stage attack algorithm
3. `masking_threshold.py`: Calculates psychoacoustic masking for imperceptibility

**Attack Flow:**
- Stage 1: Optimize perturbations to change ASR transcription to target phrase
- Stage 2: Apply psychoacoustic masking to make perturbations imperceptible
- The attack uses gradient-based optimization with CTC loss (for wav2vec2) or cross-entropy loss (for Whisper)

**Key Design Patterns:**
- Device-agnostic code (automatic GPU/CPU detection)
- Batch processing for efficiency
- Modular component design for easy model swapping
- Memory-efficient audio processing

### Original Implementation Structure

The original implementation uses TensorFlow 1.x and the Lingvo ASR system:

**Core Components:**
- `generate_imperceptible_adv.py`: Main attack orchestration
- `generate_masking_threshold.py`: Frequency masking calculations
- `tool.py`: Utility functions for audio and model interfacing
- `room_simulator.py`: Simulates acoustic environments for robust attacks

**Attack Variants:**
- Imperceptible attacks: Optimized for minimal perceptual distortion
- Robust attacks: Designed to survive over-the-air playback in various acoustic environments

### Audio Data

Both implementations use the LibriSpeech test-clean dataset. The repository includes 10 sample audio files in `adversarial_asr/LibriSpeech/test-clean/`. Target transcriptions are defined in:
- `adversarial_asr/read_data.txt`: 10 test samples
- `adversarial_asr/util/read_data_full.txt`: Full 1000 sample list

### Model Architectures

**Modern:**
- Whisper: Transformer-based encoder-decoder ASR model
- wav2vec2: Self-supervised speech representation model with CTC decoding

**Original:**
- Lingvo: LSTM-based sequence-to-sequence ASR model with attention

## Important Considerations

1. **Security Context**: This is adversarial security research code for defensive purposes. The attacks demonstrate vulnerabilities in ASR systems to help develop more robust models.

2. **GPU Memory**: Adversarial optimization can be memory-intensive. Adjust batch sizes if encountering OOM errors.

3. **Audio Quality**: The success of attacks depends on careful tuning of perturbation bounds and masking thresholds. Different audio samples may require different parameters.

4. **Dependencies**: The modern implementation uses UV for dependency management, ensuring reproducible environments. The original requires specific versions of TensorFlow (1.13) and manual Lingvo compilation.

5. **Output Files**: 
   - Stage 1 outputs: Basic adversarial examples (suffix `_stage1.wav`)
   - Stage 2 outputs: Imperceptible adversarial examples (suffix `_stage2.wav`)
- always use uv to run python code and commands