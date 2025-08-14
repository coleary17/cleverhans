#!/bin/bash
# AWS GPU deployment script for adversarial ASR attacks
# Optimized for NVIDIA GPU instances (p3, g4dn, etc.)

set -e

echo "============================================"
echo "Adversarial ASR - AWS GPU Deployment"
echo "============================================"

# Configuration
GPU_IMAGE_NAME="adversarial-asr-gpu"
OUTPUT_DIR="./output"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
NUM_ITER_STAGE1="${NUM_ITER_STAGE1:-1000}"
NUM_ITER_STAGE2="${NUM_ITER_STAGE2:-10}"
BATCH_SIZE="${BATCH_SIZE:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# Check if running on AWS with GPU
if ! nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure you're on a GPU instance."
    echo "Recommended AWS instances: p3.2xlarge, g4dn.xlarge, or similar"
    echo ""
fi

# Build GPU Docker image
echo ""
echo "Building GPU-optimized Docker image..."
docker build -f Dockerfile.gpu -t ${GPU_IMAGE_NAME} .

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run with GPU support
echo ""
echo "Running adversarial attack with GPU acceleration..."
echo "Configuration:"
echo "  - Examples: ${NUM_EXAMPLES}"
echo "  - Stage 1 iterations: ${NUM_ITER_STAGE1}"
echo "  - Stage 2 iterations: ${NUM_ITER_STAGE2}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Device: CUDA (GPU)"
echo ""

# Mount the original data file if it exists
MOUNT_OPTS=""
if [ -f "../adversarial_asr/util/read_data_full.txt" ]; then
    echo "Mounting read_data_full.txt for full dataset..."
    MOUNT_OPTS="-v $(pwd)/../adversarial_asr/util/read_data_full.txt:/app/read_data_full.txt:ro"
fi

# Note: LibriSpeech is now downloaded in the Docker image
# No need to mount external volumes for audio data
docker run --rm \
  --gpus all \
  -v "$(pwd)/output:/app/output" \
  ${MOUNT_OPTS} \
  ${GPU_IMAGE_NAME} \
  bash -c "
    # Check if we have the full data file
    if [ -f read_data_full.txt ]; then
      echo 'Converting full dataset (1000 files) to FLAC paths...'
      uv run python convert_full_data.py --format csv
      
      # Create subset if needed
      if [ ${NUM_EXAMPLES} -lt 1000 ]; then
        echo \"Creating subset with ${NUM_EXAMPLES} examples...\"
        head -${NUM_EXAMPLES} full_data_flac.txt > attack_data.txt
      else
        echo 'Using full dataset (1000 examples)...'
        cp full_data_flac.txt attack_data.txt
      fi
    else
      echo 'Creating data file from LibriSpeech FLAC files...'
      uv run python create_flac_data.py --num ${NUM_EXAMPLES} --output attack_data.txt
    fi
    
    # Check GPU availability
    echo ''
    echo 'GPU Status:'
    uv run python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\")'
    echo ''
    
    # Run attack using the attack data file
    echo 'Starting adversarial attack on GPU...'
    uv run python -c \"
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

attack = AdversarialAttack(
    model_name='openai/whisper-base',
    device='cuda',
    batch_size=${BATCH_SIZE},
    initial_bound=0.15,
    lr_stage1=0.1,
    lr_stage2=0.01,
    num_iter_stage1=${NUM_ITER_STAGE1},
    num_iter_stage2=${NUM_ITER_STAGE2},
    log_interval=${LOG_INTERVAL},
    verbose=False
)

attack.run_attack(
    data_file='attack_data.txt',
    root_dir='.',
    output_dir='./output'
)
\"
  "

echo ""
echo "============================================"
echo "GPU attack complete!"
echo "Output files saved to: ${OUTPUT_DIR}/"
echo "============================================"

# List generated files
echo ""
echo "Generated adversarial audio files:"
ls -la ${OUTPUT_DIR}/*.wav 2>/dev/null || echo "No output files found"

# Optional: sync to S3 if AWS CLI is configured
if command -v aws &> /dev/null && [ ! -z "${S3_BUCKET}" ]; then
    echo ""
    echo "Uploading results to S3..."
    aws s3 sync ${OUTPUT_DIR}/ s3://${S3_BUCKET}/adversarial-outputs/
    echo "Results uploaded to: s3://${S3_BUCKET}/adversarial-outputs/"
fi