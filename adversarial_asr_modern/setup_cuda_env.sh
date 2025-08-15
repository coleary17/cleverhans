#!/bin/bash
# Complete setup for CUDA environment with PyTorch

set -e

echo "============================================"
echo "Complete CUDA Environment Setup"
echo "============================================"
echo ""

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "----------------------------------------"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
fi

if ! nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected!"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
echo "✅ GPU: $GPU_INFO"

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
echo "✅ CUDA Version: $CUDA_VERSION"
echo ""

# Step 2: Clean environment
echo "Step 2: Cleaning existing environment..."
echo "----------------------------------------"
rm -rf .venv
rm -rf __pycache__
rm -rf src/*.egg-info
uv cache clean 2>/dev/null || true
echo "✅ Cleaned"
echo ""

# Step 3: Determine CUDA version for PyTorch
echo "Step 3: Selecting PyTorch CUDA version..."
echo "----------------------------------------"
if [[ "$CUDA_VERSION" == "12."* ]]; then
    PYTORCH_CUDA="cu121"
    echo "Selected: CUDA 12.1 (cu121)"
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    PYTORCH_CUDA="cu118"
    echo "Selected: CUDA 11.8 (cu118)"
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    PYTORCH_CUDA="cu118"
    echo "Selected: CUDA 11.8 (cu118)"
else
    PYTORCH_CUDA="cu118"
    echo "Selected: CUDA 11.8 (default)"
fi
echo ""

# Step 4: Create new environment
echo "Step 4: Creating new virtual environment..."
echo "----------------------------------------"
uv venv --python 3.10
echo "✅ Virtual environment created"
echo ""

# Step 5: Install PyTorch with CUDA
echo "Step 5: Installing PyTorch with CUDA support..."
echo "----------------------------------------"
echo "Installing from: https://download.pytorch.org/whl/${PYTORCH_CUDA}"

# Install PyTorch first with correct CUDA
uv pip install \
    torch==2.5.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

echo "✅ PyTorch installed"
echo ""

# Step 6: Install other dependencies
echo "Step 6: Installing other dependencies..."
echo "----------------------------------------"
uv pip install \
    transformers>=4.30.0 \
    datasets>=2.12.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    openai-whisper>=20230314 \
    matplotlib>=3.7.0 \
    absl-py>=1.4.0 \
    pandas \
    audioread

echo "✅ Dependencies installed"
echo ""

# Step 7: Verify installation
echo "Step 7: Verifying CUDA installation..."
echo "----------------------------------------"

VERIFICATION=$(uv run python -c "
import torch
import sys

cuda_available = torch.cuda.is_available()
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {cuda_available}')

if cuda_available:
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    
    # Test CUDA
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print('CUDA test: PASSED')
    print('SUCCESS')
else:
    print('FAILED')
    sys.exit(1)
" 2>&1)

echo "$VERIFICATION"
echo ""

if echo "$VERIFICATION" | grep -q "SUCCESS"; then
    echo "============================================"
    echo "✅ CUDA SETUP SUCCESSFUL!"
    echo "============================================"
    echo ""
    echo "You can now run:"
    echo "  ./gpu_optimized.sh quick    # Quick test"
    echo "  ./gpu_native.sh fast        # Fast mode"
    echo ""
    echo "To activate this environment manually:"
    echo "  source .venv/bin/activate"
else
    echo "============================================"
    echo "❌ CUDA SETUP FAILED"
    echo "============================================"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check CUDA installation: nvcc --version"
    echo "2. Check LD_LIBRARY_PATH: echo \$LD_LIBRARY_PATH"
    echo "3. Try manual install:"
    echo "   uv pip install torch --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}"
    exit 1
fi