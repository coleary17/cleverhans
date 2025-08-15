#!/bin/bash
# Optimized GPU script for faster adversarial attacks
# Includes performance optimizations and monitoring

set -e

echo "============================================"
echo "OPTIMIZED GPU Adversarial Attack"
echo "============================================"
echo ""

# Configuration
OUTPUT_DIR="./output_gpu_optimized"
RESULTS_DIR="./results_gpu_optimized"
BATCH_SIZE="${BATCH_SIZE:-30}"  # Larger batch size for GPU
LOG_INTERVAL="${LOG_INTERVAL:-50}"  # Less frequent logging

# Check for NVIDIA GPU
if nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    echo ""
else
    echo "⚠️  WARNING: No GPU detected. This script requires GPU."
    exit 1
fi

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}

# Test mode
MODE="${1:-quick}"

case $MODE in
    "quick")
        echo "Running QUICK test (5 examples, minimal iterations)..."
        NUM_EXAMPLES=5
        NUM_ITER_STAGE1=50
        NUM_ITER_STAGE2=10
        BATCH_SIZE=5
        ;;
    "fast")
        echo "Running FAST test (20 examples, reduced iterations)..."
        NUM_EXAMPLES=20
        NUM_ITER_STAGE1=200
        NUM_ITER_STAGE2=50
        BATCH_SIZE=20
        ;;
    "standard")
        echo "Running STANDARD test (100 examples, normal iterations)..."
        NUM_EXAMPLES=100
        NUM_ITER_STAGE1=500
        NUM_ITER_STAGE2=100
        BATCH_SIZE=25
        ;;
    "full")
        echo "Running FULL test (1000 examples, full iterations)..."
        read -p "This will take hours. Continue? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            exit 0
        fi
        NUM_EXAMPLES=1000
        NUM_ITER_STAGE1=1000
        NUM_ITER_STAGE2=200
        BATCH_SIZE=50
        ;;
    *)
        echo "Usage: $0 [quick|fast|standard|full]"
        exit 1
        ;;
esac

echo ""
echo "Configuration:"
echo "  - Examples: ${NUM_EXAMPLES}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Stage 1 iterations: ${NUM_ITER_STAGE1}"
echo "  - Stage 2 iterations: ${NUM_ITER_STAGE2}"
echo "  - Log interval: ${LOG_INTERVAL}"
echo ""

# Monitor GPU usage in background
(while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> gpu_usage.log
    sleep 5
done) &
GPU_MONITOR_PID=$!

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $GPU_MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT

# Ensure LibriSpeech is available
if [ ! -d "LibriSpeech/test-clean" ]; then
    echo "Downloading LibriSpeech dataset..."
    ./download_librispeech.sh || {
        echo "Failed to download LibriSpeech"
        exit 1
    }
fi

# Prepare data file
if [ -f "full_data_flac.txt" ]; then
    head -${NUM_EXAMPLES} full_data_flac.txt > attack_data_optimized.txt
    DATA_FILE=attack_data_optimized.txt
else
    echo "Creating data file..."
    python create_flac_data.py --num ${NUM_EXAMPLES} --output attack_data_optimized.txt
    DATA_FILE=attack_data_optimized.txt
fi

echo "Using data file: ${DATA_FILE}"
echo "First example:"
head -1 ${DATA_FILE}
echo ""

# Run optimized attack
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${MODE}_${TIMESTAMP}.csv"

echo "=== Starting Optimized Attack ==="
echo ""

# Set environment variables for optimization
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run with timing
START_TIME=$(date +%s)

uv run python -c "
import sys
import torch
import time
from pathlib import Path
sys.path.insert(0, 'src')

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

print('🚀 Initializing OPTIMIZED attack...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

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
    verbose=False,
    save_audio=False,  # Disable to save time
    skip_stage2_on_failure=True  # Skip Stage 2 for failed Stage 1 attacks
)

print('')
print('⚡ Running optimized attack...')
start = time.time()

attack.run_attack(
    data_file='${DATA_FILE}',
    root_dir='.',
    output_dir='${OUTPUT_DIR}',
    results_file='${RESULTS_FILE}'
)

elapsed = time.time() - start
print(f'')
print(f'✅ Attack completed in {elapsed/60:.1f} minutes')
print(f'Average time per example: {elapsed/${NUM_EXAMPLES}:.1f} seconds')

# Show results summary
try:
    import pandas as pd
    df = pd.read_csv('${RESULTS_FILE}')
    success_rate = df['success'].mean() * 100 if 'success' in df.columns else 0
    print(f'Success rate: {success_rate:.1f}%')
except:
    pass
"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "OPTIMIZED Attack Complete!"
echo "============================================"
echo "Total time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
echo "Results: ${RESULTS_FILE}"

# Show GPU usage summary
if [ -f gpu_usage.log ]; then
    echo ""
    echo "GPU Usage Summary:"
    awk -F',' '{sum+=$2; count++} END {print "  Average GPU Utilization: " sum/count "%"}' gpu_usage.log
    rm gpu_usage.log
fi

echo ""
echo "Performance tips:"
echo "1. Increase batch_size if GPU memory allows: BATCH_SIZE=50 $0 $MODE"
echo "2. Reduce iterations for faster testing: NUM_ITER_STAGE1=100 $0 $MODE"
echo "3. Monitor GPU: watch -n 1 nvidia-smi"
echo "============================================"