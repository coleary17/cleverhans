#!/usr/bin/env python
"""
Test basic components of the adversarial ASR system to see what works
with modern Python without the full Lingvo system.
"""

import os
import sys
import numpy as np

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Testing basic components...\n")

# Test 1: Check if we can import basic dependencies
print("=== Testing Basic Dependencies ===")
try:
    import scipy.io.wavfile as wav
    print("✓ scipy.io.wavfile imported successfully")
except ImportError as e:
    print("✗ Failed to import scipy.io.wavfile:", e)

try:
    import librosa
    print("✓ librosa imported successfully")
    print("  librosa version:", librosa.__version__)
except ImportError as e:
    print("✗ Failed to import librosa:", e)

try:
    import numpy as np
    print("✓ numpy imported successfully")
    print("  numpy version:", np.__version__)
except ImportError as e:
    print("✗ Failed to import numpy:", e)

# Test 2: Check if we can load the sample audio
print("\n=== Testing Audio Loading ===")
audio_file = "LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
if os.path.exists(audio_file):
    print(f"✓ Audio file exists: {audio_file}")
    try:
        sample_rate, audio_data = wav.read(audio_file)
        print(f"✓ Audio loaded successfully:")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Audio shape: {audio_data.shape}")
        print(f"  Audio dtype: {audio_data.dtype}")
        print(f"  Audio range: [{np.min(audio_data)}, {np.max(audio_data)}]")
    except Exception as e:
        print("✗ Failed to load audio:", e)
else:
    print(f"✗ Audio file not found: {audio_file}")

# Test 3: Try importing the masking threshold module
print("\n=== Testing Masking Threshold Module ===")
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import generate_masking_threshold
    print("✓ generate_masking_threshold imported successfully")
    
    # Test if we can call the basic function
    if os.path.exists(audio_file):
        try:
            sample_rate, audio_data = wav.read(audio_file)
            # Convert to float if needed
            if audio_data.dtype != np.float64:
                if np.max(audio_data) > 1:
                    audio_float = audio_data.astype(np.float64) / 32768.0
                else:
                    audio_float = audio_data.astype(np.float64)
            else:
                audio_float = audio_data
                
            print("  Testing masking threshold computation...")
            theta_xs, psd_max = generate_masking_threshold.generate_th(audio_float, sample_rate)
            print(f"✓ Masking threshold computed successfully")
            print(f"  theta_xs shape: {np.array(theta_xs).shape}")
            print(f"  psd_max: {psd_max}")
        except Exception as e:
            print("✗ Failed to compute masking threshold:", e)
            import traceback
            traceback.print_exc()
            
except ImportError as e:
    print("✗ Failed to import generate_masking_threshold:", e)

# Test 4: Check model checkpoint
print("\n=== Testing Model Checkpoint ===")
checkpoint_files = [
    "model/ckpt-00908156.data-00000-of-00001",
    "model/ckpt-00908156.index", 
    "model/ckpt-00908156.meta"
]

for checkpoint_file in checkpoint_files:
    if os.path.exists(checkpoint_file):
        size = os.path.getsize(checkpoint_file)
        print(f"✓ {checkpoint_file} exists ({size:,} bytes)")
    else:
        print(f"✗ {checkpoint_file} not found")

# Test 5: Check data file
print("\n=== Testing Data File ===")
if os.path.exists("read_data.txt"):
    print("✓ read_data.txt exists")
    with open("read_data.txt", 'r') as f:
        lines = f.readlines()
        print(f"  Number of lines: {len(lines)}")
        if len(lines) >= 2:
            print(f"  Sample audio paths: {lines[0].strip()[:100]}...")
            print(f"  Sample transcriptions: {lines[1].strip()[:100]}...")
else:
    print("✗ read_data.txt not found")

print("\n=== Summary ===")
print("This test verifies which components work with the current Python setup.")
print("Next steps will depend on what we can successfully import and run.")
