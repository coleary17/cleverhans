#!/bin/bash
# Use Lambda Labs pre-installed environment

echo "============================================"
echo "Lambda Labs Environment Setup"
echo "============================================"
echo ""

# Check what's already installed
echo "Checking Lambda Labs pre-installed environment..."
echo "----------------------------------------"

# Check if conda is available (Lambda often uses conda)
if command -v conda &> /dev/null; then
    echo "✅ Conda found"
    conda info --envs
    echo ""
fi

# Check system Python for PyTorch
echo "Checking system Python for PyTorch..."
python3 -c "
import sys
print(f'Python: {sys.version}')
print(f'Path: {sys.executable}')
try:
    import torch
    print(f'✅ PyTorch found: {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('❌ PyTorch not found in system Python')
" 2>/dev/null

echo ""

# Check if there's a pre-configured virtual environment
if [ -d "/opt/conda" ]; then
    echo "Found conda at /opt/conda"
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
fi

# Option 1: Use system packages directly
echo "Option 1: Creating venv with system packages..."
echo "----------------------------------------"

# Remove old venv
rm -rf .venv

# Create venv that can access system packages
python3 -m venv .venv --system-site-packages

# Activate it
source .venv/bin/activate

# Check what we have
echo "Checking available packages..."
python -c "
import importlib
packages = ['torch', 'transformers', 'numpy', 'scipy']
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: not found')

# Special check for CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA: {torch.version.cuda}')
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('❌ CUDA not available')
except:
    pass
"

echo ""
echo "Installing missing packages..."
# Only install what's missing
pip install --upgrade \
    transformers \
    datasets \
    librosa \
    soundfile \
    openai-whisper \
    audioread \
    pandas \
    absl-py \
    2>/dev/null

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"

# Final verification
python -c "
import torch
if torch.cuda.is_available():
    print('✅ CUDA WORKING!')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available')
"

echo ""
echo "To use this environment:"
echo "  source .venv/bin/activate"
echo "  python run_attack.py"
echo ""
echo "Or run directly:"
echo "  .venv/bin/python run_attack.py"