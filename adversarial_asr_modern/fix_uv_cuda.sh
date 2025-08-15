#!/bin/bash
# Fix PyTorch CUDA in uv virtual environment

set -e

echo "============================================"
echo "Fixing PyTorch CUDA in UV Environment"
echo "============================================"
echo ""

# Check if in uv project
if [ ! -f "pyproject.toml" ]; then
    echo "❌ No pyproject.toml found. Are you in the right directory?"
    exit 1
fi

# Get CUDA version
if nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "✅ Detected CUDA version: $CUDA_VERSION"
else
    echo "❌ No GPU detected"
    exit 1
fi

# Determine PyTorch CUDA package
if [[ "$CUDA_VERSION" == "12."* ]]; then
    TORCH_CUDA="cu121"
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    TORCH_CUDA="cu118"
else
    TORCH_CUDA="cu118"  # Default
fi

echo ""
echo "Removing existing PyTorch and recreating environment..."
echo "----------------------------------------"

# Remove and recreate the virtual environment
rm -rf .venv

# Create new environment with correct PyTorch
echo "Creating fresh environment with CUDA support..."
uv venv

# Activate and install with CUDA support
echo ""
echo "Installing PyTorch with CUDA ${TORCH_CUDA}..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# Install other dependencies
echo ""
echo "Installing other dependencies..."
uv sync

# Verify
echo ""
echo "Verifying installation..."
echo "----------------------------------------"

uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✅ SUCCESS!')
else:
    print('❌ Still not working')
    print('Version string:', torch.__version__)
    if '+cpu' in torch.__version__:
        print('ERROR: CPU-only version installed!')
"

echo ""
echo "============================================"
echo "Done! Test with: uv run python -c 'import torch; print(torch.cuda.is_available())'"
echo "============================================"