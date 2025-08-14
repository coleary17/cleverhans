#!/usr/bin/env python3
"""
GPU/CUDA diagnostic script for troubleshooting GPU detection issues.
"""

import sys
import subprocess
import os

print("=" * 60)
print("GPU/CUDA Diagnostic Report")
print("=" * 60)
print()

# 1. Check nvidia-smi
print("1. NVIDIA Driver Check:")
print("-" * 40)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ nvidia-smi found")
        # Get basic GPU info
        gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 
                                capture_output=True, text=True)
        print(f"   GPU: {gpu_info.stdout.strip()}")
    else:
        print("❌ nvidia-smi not found or failed")
except FileNotFoundError:
    print("❌ nvidia-smi command not found")
print()

# 2. Check CUDA environment variables
print("2. CUDA Environment Variables:")
print("-" * 40)
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'LD_LIBRARY_PATH', 'PATH']
for var in cuda_vars:
    value = os.environ.get(var, 'Not set')
    if var == 'LD_LIBRARY_PATH' or var == 'PATH':
        # Check if contains cuda
        if 'cuda' in value.lower():
            print(f"✅ {var}: Contains CUDA paths")
        else:
            print(f"⚠️  {var}: No CUDA paths found")
    else:
        print(f"   {var}: {value}")
print()

# 3. Check PyTorch
print("3. PyTorch CUDA Support:")
print("-" * 40)
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version (PyTorch built with): {torch.version.cuda if torch.version.cuda else 'None'}")
    
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        print(f"   Device capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("❌ CUDA not available in PyTorch")
        
        # Try to diagnose why
        print("\n   Possible reasons:")
        if not torch.version.cuda:
            print("   - PyTorch installed without CUDA support (CPU-only version)")
            print("   - Solution: Install PyTorch with CUDA support:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        else:
            print(f"   - PyTorch built with CUDA {torch.version.cuda} but runtime CUDA not found")
            print("   - Check CUDA installation and LD_LIBRARY_PATH")
            
except ImportError as e:
    print(f"❌ PyTorch not installed or import error: {e}")
print()

# 4. Check CUDA toolkit
print("4. CUDA Toolkit Check:")
print("-" * 40)
try:
    nvcc_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if nvcc_result.returncode == 0:
        # Parse version from output
        lines = nvcc_result.stdout.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"✅ NVCC found: {line.strip()}")
                break
    else:
        print("❌ nvcc not found or failed")
except FileNotFoundError:
    print("❌ nvcc command not found (CUDA toolkit may not be installed)")
print()

# 5. Check for common CUDA libraries
print("5. CUDA Libraries Check:")
print("-" * 40)
cuda_lib_paths = [
    '/usr/local/cuda/lib64',
    '/usr/local/cuda-11.8/lib64',
    '/usr/local/cuda-11.7/lib64',
    '/usr/local/cuda-12.0/lib64',
    '/usr/lib/x86_64-linux-gnu',
]

found_cuda_libs = False
for path in cuda_lib_paths:
    if os.path.exists(path):
        # Check for libcudart
        cudart_path = os.path.join(path, 'libcudart.so')
        if os.path.exists(cudart_path) or os.path.exists(cudart_path + '.11.0'):
            print(f"✅ CUDA libraries found in: {path}")
            found_cuda_libs = True
            break

if not found_cuda_libs:
    print("❌ CUDA libraries not found in standard locations")
print()

# 6. Recommendations
print("=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

try:
    import torch
    if not torch.cuda.is_available():
        if not torch.version.cuda:
            print("1. Install PyTorch with CUDA support:")
            print("   # For CUDA 11.8:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print()
            print("   # For CUDA 12.1:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print()
            print("   # Using uv:")
            print("   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("1. PyTorch has CUDA support but can't find CUDA runtime")
            print("   - Verify CUDA installation")
            print("   - Set LD_LIBRARY_PATH:")
            print("     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
            print("   - Ensure CUDA version matches PyTorch's CUDA version")
    else:
        print("✅ Everything looks good! CUDA is available.")
except:
    print("1. Install PyTorch first")

print()
print("=" * 60)