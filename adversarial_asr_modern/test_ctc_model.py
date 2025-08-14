#!/usr/bin/env python3
"""
Test script to verify CTC model functionality.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.ctc_audio_utils import test_ctc_model

def main():
    """Test CTC model with different sizes."""
    print("Testing CTC Model Functionality")
    print("=" * 50)
    
    # Test base model
    print("\n1. Testing base model...")
    try:
        test_ctc_model('base')
        print("✅ Base model test passed")
    except Exception as e:
        print(f"❌ Base model test failed: {e}")
    
    # Look for a test audio file
    test_audio = None
    for audio_path in ["../adversarial_asr/LibriSpeech/test-clean/61/70968/61-70968-0011.wav",
                       "../adversarial_asr/LibriSpeech/test-clean/2300/131720/2300-131720-0015.wav"]:
        if Path(audio_path).exists():
            test_audio = audio_path
            break
    
    if test_audio:
        print(f"\n2. Testing with actual audio file: {test_audio}")
        try:
            test_ctc_model('base', test_audio)
            print("✅ Audio transcription test passed")
        except Exception as e:
            print(f"❌ Audio transcription test failed: {e}")
    else:
        print("\n2. No test audio file found - skipping audio test")
    
    print("\nCTC model testing completed.")

if __name__ == "__main__":
    main()
