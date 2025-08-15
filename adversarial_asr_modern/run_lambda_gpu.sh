#!/bin/bash
# Run attack on Lambda Labs GPU using system packages

set -e

echo "============================================"
echo "Lambda Labs GPU Attack Runner"
echo "============================================"
echo ""

# Configuration
OUTPUT_DIR="./output_lambda"
RESULTS_DIR="./results_lambda"
MODE="${1:-quick}"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
echo ""

# Check PyTorch in system
echo "Checking system PyTorch..."
PYTORCH_CHECK=$(python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" 2>&1)

echo "$PYTORCH_CHECK"

if ! echo "$PYTORCH_CHECK" | grep -q "CUDA: True"; then
    echo ""
    echo "⚠️  WARNING: CUDA not detected in system Python"
    echo "Trying to find working Python environment..."
    
    # Try different Python paths
    for PYTHON in /usr/bin/python3 /opt/conda/bin/python python3; do
        if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "✅ Found working Python: $PYTHON"
            PYTHON_CMD=$PYTHON
            break
        fi
    done
    
    if [ -z "$PYTHON_CMD" ]; then
        echo "❌ No Python with CUDA found. Using default."
        PYTHON_CMD=python3
    fi
else
    PYTHON_CMD=python3
fi

echo ""
echo "Using Python: $PYTHON_CMD"
echo ""

# Set parameters based on mode
case $MODE in
    "quick")
        NUM_EXAMPLES=5
        NUM_ITER=50
        BATCH_SIZE=5
        ;;
    "fast")
        NUM_EXAMPLES=20
        NUM_ITER=200
        BATCH_SIZE=10
        ;;
    "full")
        NUM_EXAMPLES=100
        NUM_ITER=500
        BATCH_SIZE=20
        ;;
    *)
        echo "Usage: $0 [quick|fast|full]"
        exit 1
        ;;
esac

echo "Mode: $MODE"
echo "Examples: $NUM_EXAMPLES"
echo "Iterations: $NUM_ITER"
echo "Batch size: $BATCH_SIZE"
echo ""

# Download LibriSpeech if needed
if [ ! -d "LibriSpeech/test-clean" ]; then
    echo "Downloading LibriSpeech..."
    wget -q --show-progress http://www.openslr.org/resources/12/test-clean.tar.gz
    tar -xzf test-clean.tar.gz
    rm test-clean.tar.gz
fi

# Prepare data file
if [ -f "full_data_flac.txt" ]; then
    head -${NUM_EXAMPLES} full_data_flac.txt > attack_data_lambda.txt
else
    # Create simple data file
    echo "Creating data file..."
    find LibriSpeech/test-clean -name "*.flac" | head -${NUM_EXAMPLES} > temp_files.txt
    while read -r file; do
        echo "$file,original text,hello world" >> attack_data_lambda.txt
    done < temp_files.txt
    rm temp_files.txt
fi

DATA_FILE=attack_data_lambda.txt
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${MODE}_${TIMESTAMP}.csv"

# Install minimal requirements if needed
echo "Installing required packages..."
$PYTHON_CMD -m pip install --quiet \
    transformers \
    openai-whisper \
    librosa \
    soundfile \
    audioread \
    pandas \
    2>/dev/null || true

echo ""
echo "Starting attack..."
echo "============================================"

# Run the attack
$PYTHON_CMD << EOF
import sys
import os
import time
sys.path.insert(0, 'src')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import torch
print(f"Using PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = 'cuda'
else:
    print("WARNING: Using CPU")
    device = 'cpu'

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

print("\nInitializing attack...")
attack = AdversarialAttack(
    model_name='openai/whisper-base',
    device=device,
    batch_size=${BATCH_SIZE},
    initial_bound=0.15,
    lr_stage1=0.1,
    lr_stage2=0.01,
    num_iter_stage1=${NUM_ITER},
    num_iter_stage2=50,
    log_interval=50,
    verbose=False,
    save_audio=False
)

print("Running attack...")
start_time = time.time()

attack.run_attack(
    data_file='${DATA_FILE}',
    root_dir='.',
    output_dir='${OUTPUT_DIR}',
    results_file='${RESULTS_FILE}'
)

elapsed = time.time() - start_time
print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
print(f"Results saved to: ${RESULTS_FILE}")

# Show summary
try:
    import pandas as pd
    df = pd.read_csv('${RESULTS_FILE}')
    if 'success' in df.columns:
        print(f"Success rate: {df['success'].mean()*100:.1f}%")
except:
    pass
EOF

echo ""
echo "============================================"
echo "Attack Complete!"
echo "============================================"
echo "Results: ${RESULTS_FILE}"

# Check if results exist
if [ -f "$RESULTS_FILE" ]; then
    echo "✅ Results file created"
    wc -l "$RESULTS_FILE"
else
    echo "⚠️  No results file found"
fi

echo ""
echo "Next steps:"
echo "1. Check results: cat $RESULTS_FILE"
echo "2. Run larger test: $0 fast"
echo "3. Download results: scp ubuntu@[instance-ip]:$(pwd)/$RESULTS_FILE ./"
echo "============================================"