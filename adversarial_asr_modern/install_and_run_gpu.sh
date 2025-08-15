#!/bin/bash
# Install UV and run GPU optimized attack without Docker

set -e

echo "============================================"
echo "Direct GPU Attack Setup"
echo "============================================"
echo ""

# Check GPU
if nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️ WARNING: No GPU detected"
fi

# Install UV if not present
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the env file to add UV to PATH
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    # Add to current PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    # Verify UV is available
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install UV. Trying alternative method..."
        pip install uv
    fi
fi

echo "UV location: $(which uv)"
uv --version
echo ""

# Install dependencies
echo "Installing dependencies..."
uv venv
uv pip install -r requirements.txt

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA..."

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
echo "CUDA Version: $CUDA_VERSION"

# Install appropriate PyTorch
if [[ "$CUDA_VERSION" == "12."* ]]; then
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    uv pip install torch torchvision torchaudio
fi

# Verify CUDA
echo ""
echo "Verifying PyTorch CUDA..."
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✅ CUDA working!')
"

echo ""
echo "Setup complete! Now running attack..."
echo "============================================"
echo ""

# Run the optimized attack
exec ./gpu_optimized.sh quick
