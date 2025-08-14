#!/bin/bash
# GPU test script for full LibriSpeech dataset attack
# Native version (no Docker) - optimized for Lambda Labs GPU instances

set -e

echo "============================================"
echo "GPU Full Dataset Test - Adversarial ASR"
echo "============================================"
echo ""

# Configuration
OUTPUT_DIR="./output_gpu"
RESULTS_DIR="./results_gpu"

# Check for NVIDIA GPU
if nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  WARNING: No GPU detected. This will be slow on CPU."
    echo ""
fi

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}

# Test modes
MODE="${1:-quick}"  # quick, subset, or full

case $MODE in
    "quick")
        echo "Running QUICK test (1 example, 10 iterations)..."
        NUM_EXAMPLES=1
        NUM_ITER_STAGE1=10
        NUM_ITER_STAGE2=2
        ;;
    "subset")
        echo "Running SUBSET test (10 examples, 100 iterations)..."
        NUM_EXAMPLES=10
        NUM_ITER_STAGE1=100
        NUM_ITER_STAGE2=10
        ;;
    "full")
        echo "Running FULL test (1000 examples, 1000 iterations)..."
        echo "⚠️  WARNING: This will take HOURS or DAYS!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Aborted."
            exit 0
        fi
        NUM_EXAMPLES=1000
        NUM_ITER_STAGE1=1000
        NUM_ITER_STAGE2=100
        ;;
    *)
        echo "Usage: $0 [quick|subset|full]"
        exit 1
        ;;
esac

echo ""
echo "Configuration:"
echo "  - Examples: ${NUM_EXAMPLES}"
echo "  - Stage 1 iterations: ${NUM_ITER_STAGE1}"
echo "  - Stage 2 iterations: ${NUM_ITER_STAGE2}"
echo "  - Output: ${OUTPUT_DIR}"
echo ""

# Install dependencies if needed
echo "Checking Python environment..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
fi

# Install project dependencies
if [ -f "pyproject.toml" ] || [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    if [ -f "pyproject.toml" ]; then
        uv sync
    elif [ -f "requirements.txt" ]; then
        uv pip install -r requirements.txt
    fi
fi

echo ""
echo "=== Checking Environment ==="

# Check GPU availability
echo 'Checking GPU...'
python -c 'import torch; print(f"PyTorch CUDA available: {torch.cuda.is_available()}"); print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")'
echo ''

# Check LibriSpeech dataset
echo 'Checking LibriSpeech dataset...'
if [ -d "LibriSpeech/test-clean" ]; then
    FLAC_COUNT=$(find LibriSpeech/test-clean -name '*.flac' 2>/dev/null | wc -l)
    echo "Found ${FLAC_COUNT} FLAC files"
else
    echo "LibriSpeech dataset not found. Downloading..."
    echo "This will download ~350MB and may take a few minutes..."
    
    # Check if wget is available, otherwise use curl
    if command -v wget &> /dev/null; then
        # Download test-clean dataset with wget
        wget -q --show-progress http://www.openslr.org/resources/12/test-clean.tar.gz
    elif command -v curl &> /dev/null; then
        # Download with curl
        curl -L -o test-clean.tar.gz http://www.openslr.org/resources/12/test-clean.tar.gz
    else
        echo "❌ Neither wget nor curl found. Please install one of them."
        echo "   On Ubuntu/Debian: sudo apt-get install wget"
        echo "   On macOS: brew install wget"
        exit 1
    fi
    
    if [ -f "test-clean.tar.gz" ]; then
        echo "Extracting LibriSpeech test-clean..."
        tar -xzf test-clean.tar.gz
        rm test-clean.tar.gz  # Remove archive to save space
        
        # Verify extraction
        if [ -d "LibriSpeech/test-clean" ]; then
            FLAC_COUNT=$(find LibriSpeech/test-clean -name '*.flac' 2>/dev/null | wc -l)
            echo "✅ LibriSpeech downloaded successfully!"
            echo "   Found ${FLAC_COUNT} FLAC files"
        else
            echo "❌ Failed to extract LibriSpeech dataset"
            exit 1
        fi
    else
        echo "❌ Failed to download LibriSpeech dataset"
        echo "Please download manually:"
        echo "  wget http://www.openslr.org/resources/12/test-clean.tar.gz"
        echo "  tar -xzf test-clean.tar.gz"
        exit 1
    fi
fi
echo ''

# Convert the full data file to FLAC paths
echo 'Preparing data file...'
if [ -f read_data_full.txt ]; then
    echo 'Converting read_data_full.txt to FLAC paths...'
    python convert_full_data.py --format csv
    
    # Check conversion
    if [ -f full_data_flac.txt ]; then
        echo "✅ Created full_data_flac.txt with $(wc -l < full_data_flac.txt) entries"
    else
        echo '❌ Failed to create full_data_flac.txt'
        exit 1
    fi
else
    echo 'Creating data file from LibriSpeech FLAC files...'
    python create_flac_data.py --num ${NUM_EXAMPLES} --output attack_data.txt
    
    if [ -f attack_data.txt ]; then
        echo "✅ Created attack_data.txt with ${NUM_EXAMPLES} entries"
        DATA_FILE=attack_data.txt
    else
        echo '❌ Failed to create attack_data.txt'
        exit 1
    fi
fi

# Select data file
if [ -f full_data_flac.txt ]; then
    # Create subset if not using full dataset
    if [ ${NUM_EXAMPLES} -lt 1000 ]; then
        head -${NUM_EXAMPLES} full_data_flac.txt > attack_data_subset.txt
        DATA_FILE=attack_data_subset.txt
        echo "Using subset: first ${NUM_EXAMPLES} examples"
    else
        DATA_FILE=full_data_flac.txt
        echo 'Using full dataset: 1000 examples'
    fi
fi

echo ''
echo 'First 3 entries in data file:'
head -3 ${DATA_FILE} | cut -d',' -f1
echo ''

# Run the attack
echo '=== Starting Adversarial Attack ==='
echo "Data file: ${DATA_FILE}"
echo "Device: GPU (CUDA)"
echo ''

# Create results file name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/attack_results_${MODE}_${TIMESTAMP}.csv"

# Run attack with GPU
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

print('Initializing attack with GPU...')
attack = AdversarialAttack(
    model_name='openai/whisper-base',
    device='cuda',  # Force GPU
    batch_size=10,  # Larger batch for GPU
    initial_bound=0.15,
    lr_stage1=0.1,
    lr_stage2=0.01,
    num_iter_stage1=${NUM_ITER_STAGE1},
    num_iter_stage2=${NUM_ITER_STAGE2},
    log_interval=10,
    verbose=False,
    save_audio=True  # Save adversarial audio files
)

print('Running attack...')
attack.run_attack(
    data_file='${DATA_FILE}',
    root_dir='.',
    output_dir='${OUTPUT_DIR}',
    results_file='${RESULTS_FILE}'
)

print('')
print('Attack completed!')
print(f'Results saved to: ${RESULTS_FILE}')

# Show summary
try:
    import pandas as pd
    df = pd.read_csv('${RESULTS_FILE}')
    print(f'Total attacks: {len(df)}')
    if 'success' in df.columns:
        print(f'Successful: {df[\"success\"].sum()} ({100*df[\"success\"].mean():.1f}%)')
except:
    pass
"

# Check outputs
echo ''
echo '=== Checking Results ==='
if [ -f "${RESULTS_FILE}" ]; then
    echo "✅ Results file created: ${RESULTS_FILE}"
    echo "   Rows: $(wc -l < ${RESULTS_FILE})"
else
    echo '❌ No results file created'
fi

OUTPUT_COUNT=$(ls -1 ${OUTPUT_DIR}/*.wav 2>/dev/null | wc -l)
if [ ${OUTPUT_COUNT} -gt 0 ]; then
    echo "✅ Generated ${OUTPUT_COUNT} adversarial audio files"
else
    echo '⚠️  No audio files generated'
fi

echo ""
echo "============================================"
echo "GPU Test Complete!"
echo "============================================"
echo ""

# Check results
if ls ${OUTPUT_DIR}/*.wav 1> /dev/null 2>&1; then
    echo "✅ SUCCESS: Generated adversarial audio files"
    echo "   Output: ${OUTPUT_DIR}/"
    echo "   Count: $(ls -1 ${OUTPUT_DIR}/*.wav | wc -l) files"
else
    echo "⚠️  WARNING: No output files found"
fi

if ls ${RESULTS_DIR}/*.csv 1> /dev/null 2>&1; then
    echo "✅ Results saved to: ${RESULTS_DIR}/"
    latest_result=$(ls -t ${RESULTS_DIR}/*.csv | head -1)
    echo "   Latest: $(basename $latest_result)"
fi

echo ""
echo "Next steps:"
case $MODE in
    "quick")
        echo "1. Run subset test: $0 subset"
        echo "2. Check output files in ${OUTPUT_DIR}/"
        echo "3. Download results: scp -i ~/.ssh/your-key.pem ubuntu@instance-ip:$(pwd)/${OUTPUT_DIR}/*.wav ./"
        ;;
    "subset")
        echo "1. Run full test: $0 full"
        echo "2. Analyze results in ${RESULTS_DIR}/"
        echo "3. Download results: scp -i ~/.ssh/your-key.pem ubuntu@instance-ip:$(pwd)/${RESULTS_DIR}/*.csv ./"
        ;;
    "full")
        echo "1. Download results: scp -r -i ~/.ssh/your-key.pem ubuntu@instance-ip:$(pwd)/${RESULTS_DIR}/ ./"
        echo "2. Download audio: scp -r -i ~/.ssh/your-key.pem ubuntu@instance-ip:$(pwd)/${OUTPUT_DIR}/ ./"
        echo "3. Analyze attack success rates"
        ;;
esac
echo "============================================"