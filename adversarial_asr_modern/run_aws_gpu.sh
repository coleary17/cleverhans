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

# Note: LibriSpeech is now downloaded in the Docker image
# No need to mount external volumes for audio data
docker run --rm \
  --gpus all \
  -v "$(pwd)/output:/app/output" \
  ${GPU_IMAGE_NAME} \
  bash -c "
    # Create data file from FLAC files if needed
    if [ ! -f test_flac_data.txt ]; then
      echo 'Creating data file from LibriSpeech FLAC files...'
      python create_flac_data.py --num ${NUM_EXAMPLES}
    fi
    
    # Run attack on FLAC files
    uv run python run_attack.py \
      --num-examples ${NUM_EXAMPLES} \
      --num-iter-stage1 ${NUM_ITER_STAGE1} \
      --num-iter-stage2 ${NUM_ITER_STAGE2} \
      --batch-size ${BATCH_SIZE} \
      --log-interval ${LOG_INTERVAL} \
      --device cuda
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