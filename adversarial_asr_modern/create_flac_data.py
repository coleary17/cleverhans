#!/usr/bin/env python3
"""
Create data files for LibriSpeech FLAC files.
Maps FLAC files to their transcriptions and generates target phrases.
"""

import os
from pathlib import Path
import random

# Sample target phrases for adversarial attacks
TARGET_PHRASES = [
    "OK GOOGLE CALL MOM",
    "HEY SIRI SEND MONEY",
    "ALEXA BUY MORE ITEMS",
    "OPEN THE FRONT DOOR",
    "TRANSFER ALL FUNDS NOW",
    "DELETE ALL MY FILES",
    "SEND EMAIL TO BOSS",
    "UNLOCK THE CAR DOOR",
    "TURN OFF THE ALARM",
    "APPROVE THE PAYMENT NOW",
]

def read_transcript(trans_file):
    """Read LibriSpeech transcript file."""
    transcripts = {}
    with open(trans_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, text = parts
                transcripts[file_id] = text.upper()
    return transcripts

def create_data_file(librispeech_dir="LibriSpeech/test-clean", 
                     output_file="test_flac_data.txt",
                     num_examples=10):
    """
    Create a data file mapping FLAC files to transcriptions and targets.
    
    Format: audio_file,original_transcription,target_transcription
    """
    
    # Find all FLAC files
    flac_files = list(Path(librispeech_dir).glob("**/*.flac"))
    
    if not flac_files:
        print(f"No FLAC files found in {librispeech_dir}")
        print("Run ./download_librispeech.sh first")
        return
    
    print(f"Found {len(flac_files)} FLAC files")
    
    # Collect transcriptions
    all_data = []
    
    for speaker_dir in Path(librispeech_dir).iterdir():
        if not speaker_dir.is_dir():
            continue
            
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
                
            trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            if trans_file.exists():
                transcripts = read_transcript(trans_file)
                
                # Match FLAC files with transcripts
                for flac_file in chapter_dir.glob("*.flac"):
                    file_id = flac_file.stem
                    if file_id in transcripts:
                        # Use relative path from working directory
                        rel_path = flac_file.relative_to(".")
                        original_text = transcripts[file_id]
                        target_text = random.choice(TARGET_PHRASES)
                        all_data.append((str(rel_path), original_text, target_text))
    
    # Shuffle and select subset
    random.shuffle(all_data)
    selected_data = all_data[:num_examples]
    
    # Write data file
    with open(output_file, 'w') as f:
        for audio_file, original, target in selected_data:
            f.write(f"{audio_file},{original},{target}\n")
    
    print(f"Created {output_file} with {len(selected_data)} examples")
    
    # Show first few examples
    print("\nFirst 3 examples:")
    for i, (audio, orig, tgt) in enumerate(selected_data[:3]):
        print(f"{i+1}. {audio}")
        print(f"   Original: {orig[:50]}...")
        print(f"   Target:   {tgt}")

def create_full_dataset(librispeech_dir="LibriSpeech/test-clean",
                       output_file="full_flac_data.txt"):
    """Create data file with ALL LibriSpeech test-clean files."""
    
    create_data_file(librispeech_dir, output_file, num_examples=2620)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create LibriSpeech FLAC data files")
    parser.add_argument("--num", type=int, default=10,
                       help="Number of examples to include")
    parser.add_argument("--full", action="store_true",
                       help="Create full dataset with all files")
    parser.add_argument("--output", type=str, default="test_flac_data.txt",
                       help="Output data file name")
    
    args = parser.parse_args()
    
    if args.full:
        create_full_dataset(output_file=args.output)
    else:
        create_data_file(num_examples=args.num, output_file=args.output)