#!/usr/bin/env python3
"""Download and check LibriSpeech transcriptions."""

import requests
import tarfile
import os
from pathlib import Path

# First, let's check if we have the transcription files locally
transcription_files = []
librispeech_dir = Path("/Users/conor/Dev/thesis/cleverhans/adversarial_asr/LibriSpeech/test-clean")

if librispeech_dir.exists():
    # Look for transcription files
    for trans_file in librispeech_dir.rglob("*.txt"):
        transcription_files.append(trans_file)
    print(f"Found {len(transcription_files)} local transcription files")
    
    if transcription_files:
        # Read some transcriptions
        all_transcriptions = {}
        for trans_file in transcription_files[:50]:  # Check first 50 files
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_id, text = parts
                        all_transcriptions[audio_id] = text
        
        print(f"Loaded {len(all_transcriptions)} transcriptions")
        print("\nSample transcriptions:")
        for i, (audio_id, text) in enumerate(list(all_transcriptions.items())[:5]):
            print(f"{audio_id}: {text[:80]}...")
else:
    print("LibriSpeech directory not found locally")
    print("Let's download a sample of the test-clean dataset...")
    
    # Download just the transcription files
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    print(f"This would download from: {url}")
    print("The test-clean dataset is about 350MB")
    
# Now let's check against our CSV results
import pandas as pd
import sys
sys.path.append('/Users/conor/Dev/thesis/cleverhans/adversarial_asr_modern/src')

csv_path = "/Users/conor/Dev/thesis/cleverhans/adversarial_asr_modern/results_gpu_optimized/results_full_20250815_012857.csv"
df = pd.read_csv(csv_path)

# Check some specific "hallucinated" texts that seem too coherent
suspicious_texts = [
    "My last savings is an acquisition of the Sincere Valley of New Zealand",
    "Today, 6 miles away from the rainy season, the road is as good",
    "Hello boys, I'm Klaus, I'm from Israel. I'm a teacher at the U.S. University",
    "Turn up the race to end the season, and the knockoff is a cool day",
    "You're left with a good dog, and you're not gonna afford a little dream"
]

print("\n" + "="*60)
print("Checking suspicious 'hallucinated' texts...")
for text_fragment in suspicious_texts:
    found = False
    for idx, row in df.iterrows():
        # Check if this appears as an original text elsewhere
        if pd.notna(row['original_text']) and text_fragment[:30].lower() in row['original_text'].lower():
            print(f"\nFOUND: '{text_fragment[:50]}...'")
            print(f"  Appears in row {idx} as original_text")
            found = True
            break
    if not found:
        print(f"\nNOT FOUND in CSV: '{text_fragment[:50]}...'")