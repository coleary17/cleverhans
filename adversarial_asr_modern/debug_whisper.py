#!/usr/bin/env python3
"""
Debug script to test Whisper transcription on a single audio file.
"""

import sys
from pathlib import Path
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import WhisperASRModel, load_audio_file

def main():
    print("=== Whisper Debug Test ===")
    
    # Test single audio file
    audio_file = "../adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
    
    # Load audio
    print(f"Loading audio: {audio_file}")
    try:
        audio, sr = load_audio_file(audio_file, target_sr=16000)
        print(f"Audio loaded - Length: {len(audio)}, Sample rate: {sr}")
        print(f"Audio range: {np.min(audio):.4f} to {np.max(audio):.4f}")
        print(f"Audio duration: {len(audio)/sr:.2f} seconds")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Initialize Whisper
    print("\nInitializing Whisper model...")
    try:
        whisper_model = WhisperASRModel("openai/whisper-base", device='cpu')
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        return
    
    # Test transcription
    print("\nTesting transcription...")
    try:
        transcription = whisper_model.transcribe(audio, sample_rate=sr)
        print(f"Transcription: '{transcription}'")
        print(f"Length: {len(transcription)} characters")
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with different audio scaling
    print("\nTesting with different audio scaling...")
    scaled_audio = audio * 32768  # Scale to int16 range
    try:
        transcription2 = whisper_model.transcribe(scaled_audio, sample_rate=sr)
        print(f"Scaled transcription: '{transcription2}'")
    except Exception as e:
        print(f"Error with scaled audio: {e}")
    
    # Test original LibriSpeech expected transcription
    expected = "THE MORE SHE IS ENGAGED IN HER PROPER DUTIES THE LESS LEISURE WILL SHE HAVE FOR IT EVEN AS AN ACCOMPLISHMENT AND A RECREATION"
    print(f"\nExpected: '{expected}'")
    print(f"Got:      '{transcription}'")
    
    print("\nDebug complete!")

if __name__ == "__main__":
    main()
