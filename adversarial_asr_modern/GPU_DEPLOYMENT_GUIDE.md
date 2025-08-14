# GPU Deployment Guide for Adversarial ASR

## Quick Start

On your GPU-enabled VM, run:

```bash
# Clone the repository
git clone <your-repo-url>
cd cleverhans/adversarial_asr_modern

# Quick test (1 example, 10 iterations)
./test_gpu_full_dataset.sh quick

# Subset test (10 examples, 100 iterations)
./test_gpu_full_dataset.sh subset

# Full test (1000 examples, 1000 iterations) - WARNING: Takes hours!
./test_gpu_full_dataset.sh full
```

## What the Setup Does

### 1. **Dockerfile.gpu**
- Downloads LibriSpeech test-clean dataset (2620 FLAC files) during build
- Installs CUDA-enabled PyTorch
- Sets up UV package manager for dependencies
- Configures GPU environment variables

### 2. **Data Handling**
- **LibriSpeech**: Downloaded automatically in Docker container
- **Full Dataset**: If `read_data_full.txt` exists, converts it to FLAC paths
- **Random Dataset**: Otherwise creates random samples from LibriSpeech
- **Format**: CSV format with `audio_path,original_text,target_text`

### 3. **Test Scripts**

#### `test_gpu_full_dataset.sh` (Recommended)
- Three modes: `quick`, `subset`, `full`
- Automatically detects GPU
- Saves results to `results_gpu/` directory
- Saves adversarial audio to `output_gpu/` directory

#### `run_aws_gpu.sh`
- Configurable via environment variables
- Default: 10 examples, 1000 iterations
- Supports S3 upload if configured

## File Structure

```
adversarial_asr_modern/
├── Dockerfile.gpu              # GPU Docker image
├── test_gpu_full_dataset.sh   # Main test script
├── run_aws_gpu.sh             # AWS-specific script
├── create_flac_data.py        # Creates data from LibriSpeech
├── convert_full_data.py       # Converts original dataset
└── src/                       # Attack implementation
```

## Expected Output

After running the attack:

1. **Audio Files**: `output_gpu/*.wav`
   - `*_stage1.wav`: Basic adversarial examples
   - `*_stage2.wav`: Imperceptible adversarial examples

2. **Results CSV**: `results_gpu/attack_results_*.csv`
   - Contains success rates and transcriptions
   - Columns: audio_name, success, target_text, final_text, etc.

## GPU Requirements

- **Minimum**: 8GB VRAM (e.g., T4, V100)
- **Recommended**: 16GB+ VRAM for larger batches
- **Instances**: AWS p3.2xlarge, g4dn.xlarge, or equivalent

## Troubleshooting

### No GPU Detected
```bash
# Check GPU availability
nvidia-smi

# If not available, ensure you're on a GPU instance
```

### Out of Memory
- Reduce batch size in the scripts
- Default batch size is 10, try 5 or 1

### Data File Issues
- Ensure `read_data_full.txt` exists if running full dataset
- Or let the system create random samples from LibriSpeech

## Performance Expectations

| Mode | Examples | Iterations | Estimated Time |
|------|----------|------------|----------------|
| quick | 1 | 10 | ~1 minute |
| subset | 10 | 100 | ~10 minutes |
| full | 1000 | 1000 | 4-8 hours |

## Notes

- The Docker image downloads ~350MB of audio data during build
- First run will be slower due to model download (~140MB for Whisper)
- Results are saved incrementally, so you can stop and analyze partial results