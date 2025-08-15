#!/bin/bash
# Force reinstall PyTorch with CUDA support
# This script ensures the correct CUDA version is installed

set -e

echo "============================================"
echo "Force PyTorch CUDA Installation"
echo "============================================"
echo ""

# Check NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected!"
    echo "This script requires an NVIDIA GPU with drivers installed."
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ""

# Get CUDA version from nvidia-smi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
echo "CUDA Version from nvidia-smi: $CUDA_VERSION"
echo ""

# Completely remove existing PyTorch
echo "Step 1: Removing ALL existing PyTorch installations..."
echo "----------------------------------------"
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
pip3 uninstall torch torchvision torchaudio -y 2>/dev/null || true
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Clear pip cache to avoid cached CPU versions
echo ""
echo "Step 2: Clearing pip cache..."
echo "----------------------------------------"
pip cache purge 2>/dev/null || true
uv cache clean 2>/dev/null || true

# Determine the right CUDA version to install
echo ""
echo "Step 3: Installing PyTorch with CUDA support..."
echo "----------------------------------------"

if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "Installing PyTorch for CUDA 12.x..."
    CUDA_VERSION_PYTORCH="cu121"
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    CUDA_VERSION_PYTORCH="cu118"
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "Installing PyTorch for CUDA 11.x..."
    CUDA_VERSION_PYTORCH="cu118"
else
    echo "Unknown CUDA version: $CUDA_VERSION"
    echo "Defaulting to CUDA 11.8 (most compatible)..."
    CUDA_VERSION_PYTORCH="cu118"
fi

# Force install with specific CUDA version
echo "Installing torch with CUDA support (${CUDA_VERSION_PYTORCH})..."

# First try with uv
if command -v uv &> /dev/null; then
    echo "Using uv to install..."
    # Force reinstall to avoid using cached versions
    uv pip install --force-reinstall \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PYTORCH}
else
    echo "Using pip to install..."
    pip install --force-reinstall --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PYTORCH}
fi

echo ""
echo "Step 4: Verifying installation..."
echo "----------------------------------------"

# Verify the installation
python3 -c "
import torch
import sys

print('Python executable:', sys.executable)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version (PyTorch built with):', torch.version.cuda if torch.version.cuda else 'None')

if torch.cuda.is_available():
    print('Number of GPUs:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.current_device())
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('')
    print('✅ SUCCESS! PyTorch can now use CUDA!')
    
    # Test CUDA with a simple operation
    print('')
    print('Testing CUDA with tensor operation...')
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print('CUDA tensor operation successful!')
    
else:
    print('')
    print('❌ FAILED: PyTorch still cannot use CUDA')
    print('')
    print('Debugging information:')
    
    # Check if this is really a CPU-only build
    if '+cpu' in torch.__version__:
        print('ERROR: You have a CPU-only build of PyTorch!')
        print('The installation failed to get the CUDA version.')
    
    # Check for library issues
    try:
        import subprocess
        result = subprocess.run(['ldd', torch.__file__.replace('__init__.py', '_C.so')], 
                              capture_output=True, text=True)
        if 'libcuda' not in result.stdout:
            print('ERROR: PyTorch is not linked against CUDA libraries')
    except:
        pass
    
    print('')
    print('Troubleshooting steps:')
    print('1. Check LD_LIBRARY_PATH:')
    import os
    print('   ', os.environ.get('LD_LIBRARY_PATH', 'Not set'))
    print('2. Try setting:')
    print('   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH')
    print('3. Make sure CUDA toolkit is installed:')
    print('   nvcc --version')
    print('4. Try manual installation:')
    print(f'   pip install torch==2.0.1+{CUDA_VERSION_PYTORCH} -f https://download.pytorch.org/whl/torch_stable.html')
    sys.exit(1)
"

echo ""
echo "============================================"
echo "Installation complete!"
echo "You can now run: ./gpu_optimized.sh"
echo "============================================"