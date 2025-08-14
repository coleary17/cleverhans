#!/usr/bin/env python3

"""
Test and compare transcriptions of different audio outputs.
"""

import torch
from pathlib import Path
import numpy as np
from src.adversarial_asr_modern.ctc_audio_utils import CTCASRModel, load_audio_file

def test_audio_files(audio_dir: str, model_size: str = 'base'):
    """Test transcriptions of all audio files in a directory."""
    
    # Initialize model
    model = CTCASRModel(model_size=model_size, device='cpu')
    print(f"Model info: {model.get_model_info()}")
    
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"Directory {audio_dir} does not exist")
        return
    
    # Find all wav files
    wav_files = sorted(list(audio_path.glob("*.wav")))
    
    if not wav_files:
        print(f"No .wav files found in {audio_dir}")
        return
    
    print(f"\nTesting {len(wav_files)} audio files:")
    print("=" * 80)
    
    results = []
    
    for wav_file in wav_files:
        try:
            # Load audio
            audio, sr = load_audio_file(str(wav_file), target_sr=16000)
            
            # Get transcription
            transcription = model.transcribe(audio)
            
            # Calculate basic stats
            duration = len(audio) / sr
            audio_range = (np.min(audio), np.max(audio))
            
            result = {
                'file': wav_file.name,
                'transcription': transcription,
                'length': len(transcription),
                'duration': duration,
                'audio_range': audio_range
            }
            results.append(result)
            
            print(f"File: {wav_file.name}")
            print(f"  Transcription ({len(transcription)}): '{transcription}'")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Audio range: [{audio_range[0]:.3f}, {audio_range[1]:.3f}]")
            print()
            
        except Exception as e:
            print(f"Error processing {wav_file.name}: {e}")
            continue
    
    # Summary comparison
    if len(results) > 1:
        print("=" * 80)
        print("SUMMARY COMPARISON:")
        print("=" * 80)
        
        for result in results:
            print(f"{result['file']:<30} | Length: {result['length']:2d} | '{result['transcription']}'")
        
        # Find best result (longest coherent transcription)
        best_result = max(results, key=lambda x: x['length'])
        print(f"\nBest result: {best_result['file']} with {best_result['length']} characters")
    
    return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test audio transcriptions')
    parser.add_argument('--audio_dir', default='./distortion_diagnosis/conservative', 
                       help='Directory containing audio files to test')
    parser.add_argument('--model_size', default='base',
                       choices=['base', 'large', 'large-lv60'])
    
    args = parser.parse_args()
    
    print("AUDIO TRANSCRIPTION TESTING")
    print("=" * 50)
    print(f"Directory: {args.audio_dir}")
    print(f"Model: {args.model_size}")
    print("=" * 50)
    
    results = test_audio_files(args.audio_dir, args.model_size)
    
    if results:
        print(f"\nTesting completed. Processed {len(results)} files.")
    else:
        print("No files were processed successfully.")

if __name__ == "__main__":
    main()
