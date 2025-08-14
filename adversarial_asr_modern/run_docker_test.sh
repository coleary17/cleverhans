#!/bin/bash
# Quick local Docker testing script for adversarial ASR attacks
# Optimized for fast iteration testing on CPU (M-series Mac compatible)

set -e

echo "==================================="
echo "Adversarial ASR - Docker Test Run"
echo "==================================="

# Build the Docker image
echo ""
echo "Building Docker image..."
docker build -t adversarial-asr-test .

# Create output directory if it doesn't exist
mkdir -p output

# Run with minimal configuration for quick testing
echo ""
echo "Running attack with minimal configuration..."
echo "- Processing 1 example"
echo "- 100 iterations (instead of 1000)"
echo "- CPU-only execution"
echo ""

docker run --rm \
  -v "$(pwd)/../adversarial_asr/LibriSpeech:/app/LibriSpeech:ro" \
  -v "$(pwd)/output:/app/output" \
  adversarial-asr-test \
  uv run python run_attack.py \
    --num-examples 1 \
    --num-iter-stage1 100 \
    --num-iter-stage2 5 \
    --log-interval 10 \
    --device cpu

echo ""
echo "==================================="
echo "Test complete!"
echo "Check ./output/ for generated adversarial audio"
echo "==================================="

# List output files
echo ""
echo "Generated files:"
ls -la output/*.wav 2>/dev/null || echo "No output files generated yet"