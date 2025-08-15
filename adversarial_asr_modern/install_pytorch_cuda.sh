#!/bin/bash
# Simple PyTorch CUDA installation without version constraints

echo "============================================"
echo "Simple PyTorch CUDA Installation"
echo "============================================"
echo ""

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected!"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
echo "CUDA Version: $CUDA_VERSION"
echo ""

# Clean existing PyTorch
echo "Removing existing PyTorch..."
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Determine CUDA index
if [[ "$CUDA_VERSION" == "12."* ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cu121"
    echo "Using CUDA 12.1 wheels"
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cu118"
    echo "Using CUDA 11.8 wheels"
else
    INDEX_URL="https://download.pytorch.org/whl/cu118"
    echo "Defaulting to CUDA 11.8 wheels"
fi

echo ""
echo "Installing PyTorch (latest compatible version)..."
echo "----------------------------------------"

# Install without specifying version - let pip find compatible one
uv pip install torch torchvision torchaudio --index-url ${INDEX_URL}

# If that fails, try without index URL (will get CPU version but at least it works)
if [ $? -ne 0 ]; then
    echo ""
    echo "CUDA wheels not available for your platform."
    echo "Installing default PyTorch (might be CPU-only)..."
    uv pip install torch torchvision torchaudio
fi

echo ""
echo "Verifying installation..."
echo "----------------------------------------"

uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✅ CUDA working!')
else:
    print('⚠️  CUDA not available - will use CPU')
"

echo ""
echo "Done! Run your attack script:"
echo "  ./gpu_optimized.sh quick"