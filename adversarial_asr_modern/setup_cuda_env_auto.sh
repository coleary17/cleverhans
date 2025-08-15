#!/bin/bash
# Auto-detect platform and install correct PyTorch with CUDA

set -e

echo "============================================"
echo "Auto-Detect CUDA Environment Setup"
echo "============================================"
echo ""

# Step 1: Detect platform
echo "Step 1: Detecting platform..."
echo "----------------------------------------"

PLATFORM=$(python3 -c "import platform; print(platform.machine())")
OS=$(python3 -c "import platform; print(platform.system().lower())")
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "Platform: $PLATFORM"
echo "OS: $OS"
echo "Python: $PYTHON_VERSION"

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
uv cache clean 2>/dev/null || true
echo "✅ Cleaned"
echo ""

# Step 3: Create new environment
echo "Step 3: Creating virtual environment..."
echo "----------------------------------------"
uv venv
echo "✅ Created"
echo ""

# Step 4: Determine installation method
echo "Step 4: Selecting PyTorch installation method..."
echo "----------------------------------------"

# Determine CUDA version for PyTorch
if [[ "$CUDA_VERSION" == "12."* ]]; then
    PYTORCH_CUDA="cu121"
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    PYTORCH_CUDA="cu118"
else
    PYTORCH_CUDA="cu118"
fi

echo "Target CUDA: ${PYTORCH_CUDA}"

# Step 5: Install PyTorch based on platform
echo ""
echo "Step 5: Installing PyTorch..."
echo "----------------------------------------"

if [[ "$PLATFORM" == "x86_64" ]] || [[ "$PLATFORM" == "amd64" ]]; then
    echo "x86_64 platform detected - using pre-built CUDA wheels"
    
    # Don't specify exact version, let pip find compatible one
    uv pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
        
elif [[ "$PLATFORM" == "aarch64" ]] || [[ "$PLATFORM" == "arm64" ]]; then
    echo "ARM platform detected - checking for CUDA wheels..."
    
    # For ARM, we might need to use different approach
    # Try without specifying version first
    if ! uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA} 2>/dev/null; then
        echo "No pre-built CUDA wheels for ARM, trying alternative..."
        
        # Try to get any CUDA version available
        uv pip install torch torchvision torchaudio 2>/dev/null || {
            echo "Standard install failed, trying conda approach..."
            echo ""
            echo "❌ PyTorch CUDA wheels not available for ARM/aarch64"
            echo ""
            echo "For ARM/aarch64 with CUDA, you need to either:"
            echo "1. Use conda/mamba:"
            echo "   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
            echo ""
            echo "2. Build from source (advanced):"
            echo "   https://github.com/pytorch/pytorch#from-source"
            echo ""
            echo "3. Use Docker with NVIDIA runtime:"
            echo "   docker run --gpus all -it pytorch/pytorch:latest-cuda11.8-cudnn8-runtime"
            exit 1
        }
    fi
else
    echo "Unknown platform: $PLATFORM"
    echo "Attempting standard installation..."
    
    uv pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
fi

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
echo "Step 7: Verifying installation..."
echo "----------------------------------------"

uv run python -c "
import torch
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print('')
    print('✅ SUCCESS! CUDA is working!')
else:
    print('')
    print('⚠️  WARNING: CUDA not available')
    
    if '+cpu' in torch.__version__:
        print('You have CPU-only PyTorch installed')
    
    print('')
    print('This might be expected on some platforms.')
    print('You can still run the attack but it will be slower.')
"

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  ./gpu_optimized.sh quick    # Test attack"
echo "  ./gpu_native.sh fast        # Alternative script"