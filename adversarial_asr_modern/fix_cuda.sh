#!/bin/bash
# Fix PyTorch CUDA installation

echo "============================================"
echo "PyTorch CUDA Fix Script"
echo "============================================"
echo ""

# Check if nvidia-smi works
if nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "CUDA Version: $CUDA_VERSION"
else
    echo "❌ No NVIDIA GPU detected with nvidia-smi"
    echo "This script requires an NVIDIA GPU with drivers installed"
    exit 1
fi

echo ""
echo "Current PyTorch status:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version (built with): {torch.version.cuda}')" 2>/dev/null || echo "PyTorch not installed"

echo ""
echo "Reinstalling PyTorch with CUDA support..."

# Uninstall existing PyTorch
echo "Removing existing PyTorch installation..."
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install based on CUDA version
if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "Installing PyTorch for CUDA 12.1..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Unknown CUDA version: $CUDA_VERSION"
    echo "Installing PyTorch for CUDA 11.8 (most compatible)..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "Verifying installation..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('✅ SUCCESS: PyTorch can now use CUDA!')
else:
    print('❌ FAILED: PyTorch still cannot use CUDA')
    print('Run ./check_gpu.py for detailed diagnostics')
"

echo ""
echo "============================================"
echo "Done! You can now run: ./gpu_native.sh"
echo "============================================"