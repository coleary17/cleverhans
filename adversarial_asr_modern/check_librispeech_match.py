#!/usr/bin/env python3
"""Check if the 'hallucinated' texts are actually from the LibriSpeech dataset."""

import pandas as pd

# Load the CSV with results
df = pd.read_csv('/Users/conor/Dev/thesis/cleverhans/adversarial_asr_modern/results_gpu_optimized/results_full_20250815_012857.csv')

# Load the LibriSpeech data file
with open('/Users/conor/Dev/thesis/cleverhans/adversarial_asr/util/read_data_full.txt', 'r') as f:
    lines = f.readlines()

# Parse the LibriSpeech data
audio_files = lines[0].strip().split(',')
original_texts = [lines[i].strip() for i in range(1, 1001)]  # Lines 2-1001 are originals
target_texts = [lines[i].strip() for i in range(1001, 2001)]  # Lines 1002-2001 are targets

print(f"Loaded {len(audio_files)} audio files")
print(f"Loaded {len(original_texts)} original texts")
print(f"Loaded {len(target_texts)} target texts")

# Create a lookup of all LibriSpeech texts
all_librispeech_texts = set()
for text in original_texts + target_texts:
    all_librispeech_texts.add(text.lower().strip())

print(f"\nTotal unique LibriSpeech texts: {len(all_librispeech_texts)}")

# Check if the "hallucinated" final texts are actually from LibriSpeech
matches_found = 0
for idx, row in df.iterrows():
    final_text = str(row['final_text']).lower().strip()
    if pd.isna(row['final_text']) or final_text == 'nan':
        continue
    
    # Check exact match
    if final_text in all_librispeech_texts:
        matches_found += 1
        if matches_found <= 5:
            print(f"\nFOUND MATCH at row {idx}:")
            print(f"  Final text: {row['final_text'][:80]}...")
            # Find which LibriSpeech sample it matches
            for i, orig in enumerate(original_texts):
                if orig.lower().strip() == final_text:
                    print(f"  Matches original text at index {i}")
                    print(f"  Expected audio: {audio_files[i] if i < len(audio_files) else 'N/A'}")
                    print(f"  CSV audio: {row['audio_file']}")
                    break
            for i, tgt in enumerate(target_texts):
                if tgt.lower().strip() == final_text:
                    print(f"  Matches target text at index {i}")
                    break

print(f"\n" + "="*60)
print(f"TOTAL EXACT MATCHES: {matches_found} out of {len(df)} ({100*matches_found/len(df):.1f}%)")

if matches_found > 900:
    print("\nHIGH MATCH RATE! The results are likely scrambled/misaligned.")
    print("The 'final_text' appears to be legitimate LibriSpeech transcriptions,")
    print("NOT model hallucinations.")