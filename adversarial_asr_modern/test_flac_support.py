#!/usr/bin/env python3
"""
Test that FLAC files work correctly with our attack pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import WhisperASRModel, load_audio_file


def test_flac_support():
    """Test loading and processing FLAC files."""
    
    print("=" * 60)
    print("Testing FLAC File Support")
    print("=" * 60)
    
    # Check for LibriSpeech FLAC files
    flac_dir = Path("LibriSpeech/test-clean")
    
    if not flac_dir.exists():
        print("\nLibriSpeech not found. Download it first:")
        print("  ./download_librispeech.sh")
        return
    
    # Find a FLAC file
    flac_files = list(flac_dir.glob("**/*.flac"))
    
    if not flac_files:
        print("No FLAC files found!")
        return
    
    print(f"Found {len(flac_files)} FLAC files")
    
    # Test loading first FLAC file
    test_file = flac_files[0]
    print(f"\nTesting with: {test_file}")
    
    try:
        # Load FLAC file
        audio, sr = load_audio_file(str(test_file), target_sr=16000)
        print(f"✅ Loaded FLAC: shape={audio.shape}, sr={sr}")
        print(f"   Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Test with Whisper
        print("\nTesting Whisper transcription...")
        model = WhisperASRModel(device='auto')
        text = model.transcribe(audio)
        print(f"✅ Transcription: '{text}'")
        
        print("\n" + "=" * 60)
        print("SUCCESS: FLAC files work correctly!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_flac_support()