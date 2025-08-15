#!/bin/bash
# Simple setup and run script for GPU attack

set -e

echo "============================================"
echo "Setting up environment for GPU attack"
echo "============================================"
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install UV if needed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clean and create fresh environment
echo "Creating fresh environment..."
rm -rf .venv
uv venv
source .venv/bin/activate

# Install packages one by one to ensure they're all present
echo "Installing core packages..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing transformers and dependencies..."
uv pip install transformers==4.36.0
uv pip install openai-whisper
uv pip install datasets
uv pip install librosa
uv pip install soundfile
uv pip install audioread
uv pip install pandas
uv pip install scipy
uv pip install absl-py

# Verify installation
echo ""
echo "Verifying installation..."
uv run python -c "
import sys
print('Python:', sys.version)
print()

# Check all required packages
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'whisper': 'OpenAI Whisper',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'pandas': 'Pandas'
}

for module, name in packages.items():
    try:
        __import__(module)
        print(f'✅ {name} installed')
    except ImportError as e:
        print(f'❌ {name} MISSING: {e}')

# Check CUDA
import torch
print()
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ CUDA not available')

# Test Whisper import
print()
print('Testing Whisper imports...')
from transformers import WhisperProcessor, WhisperForConditionalGeneration
print('✅ Whisper models can be imported')
"

echo ""
echo "============================================"
echo "Setup complete! Running attack..."
echo "============================================"
echo ""

# Ensure LibriSpeech exists
if [ ! -d "LibriSpeech/test-clean" ]; then
    echo "Downloading LibriSpeech..."
    ./download_librispeech.sh || {
        wget -q --show-progress http://www.openslr.org/resources/12/test-clean.tar.gz
        tar -xzf test-clean.tar.gz
        rm test-clean.tar.gz
    }
fi

# Create data file if needed
if [ ! -f "full_data_flac.txt" ]; then
    echo "Creating data file..."
    uv run python create_flac_data.py --num 5 --output attack_data.txt
    DATA_FILE="attack_data.txt"
else
    head -5 full_data_flac.txt > attack_data.txt
    DATA_FILE="attack_data.txt"
fi

# Run the attack
echo "Running attack on 5 examples..."
uv run python -c "
import sys
sys.path.insert(0, 'src')

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

print('Initializing attack...')
attack = AdversarialAttack(
    model_name='openai/whisper-base',
    device='cuda',
    batch_size=5,
    initial_bound=0.15,
    lr_stage1=0.1,
    lr_stage2=0.01,
    num_iter_stage1=50,
    num_iter_stage2=10,
    log_interval=10,
    verbose=True,
    save_audio=True,
    skip_stage2_on_failure=True
)

print('Running attack...')
attack.run_attack(
    data_file='${DATA_FILE}',
    root_dir='.',
    output_dir='./output',
    results_file='./results/quick_test.csv'
)

print('✅ Attack completed!')
"

echo ""
echo "============================================"
echo "Attack finished!"
echo "Check results in ./results/quick_test.csv"
echo "============================================"
