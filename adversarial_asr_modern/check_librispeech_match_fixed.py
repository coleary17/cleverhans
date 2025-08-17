#!/usr/bin/env python3
"""Check if the 'hallucinated' texts are actually from the LibriSpeech dataset."""

import pandas as pd

# Load the CSV with results
df = pd.read_csv('/Users/conor/Dev/thesis/cleverhans/adversarial_asr_modern/results_gpu_optimized/results_full_20250815_012857.csv')

# Load the LibriSpeech data file (3 lines: audio files, original texts, target texts)
with open('/Users/conor/Dev/thesis/cleverhans/adversarial_asr/util/read_data_full.txt', 'r') as f:
    lines = f.readlines()

# Parse the LibriSpeech data
audio_files = lines[0].strip().split(',')
original_texts = lines[1].strip().split(',')
target_texts = lines[2].strip().split(',')

print(f"Loaded {len(audio_files)} audio files")
print(f"Loaded {len(original_texts)} original texts")
print(f"Loaded {len(target_texts)} target texts")

# Create a lookup of all LibriSpeech texts
all_librispeech_texts = {}
for i, text in enumerate(original_texts):
    all_librispeech_texts[text.lower().strip()] = ('original', i)
for i, text in enumerate(target_texts):
    all_librispeech_texts[text.lower().strip()] = ('target', i)

print(f"\nTotal unique LibriSpeech texts: {len(all_librispeech_texts)}")

# Check some specific "hallucinated" examples from the CSV
print("\n" + "="*60)
print("Checking specific 'hallucinated' examples:")

specific_finals = [
    "My last savings is an acquisition of the Sincere Valley of New Zealand, and I'm now about five years old, and my own family has a special identity.",
    "This is a very important thing. The first thing I do is to get the energy of the",
    "We must make sure that one of the leaders has revealed it. We must listen to the",
    "You're left with a good dog, and you're not gonna afford a little dream. You hav",
    "Turn up the race to end the season, and the knockoff is a cool day of one-time e"
]

for final in specific_finals[:3]:
    final_lower = final[:50].lower().strip()
    found = False
    for lib_text, (text_type, idx) in all_librispeech_texts.items():
        if final_lower in lib_text or lib_text[:50] in final_lower:
            print(f"\nFOUND: '{final[:50]}...'")
            print(f"  Matches {text_type} text at index {idx}")
            print(f"  Full match: {original_texts[idx] if text_type == 'original' else target_texts[idx]}")
            found = True
            break
    if not found:
        print(f"\nNOT FOUND: '{final[:50]}...'")

# Check if the final texts in CSV match LibriSpeech
print("\n" + "="*60)
print("Checking all final_text values in CSV:")

matches_found = 0
misalignment_pattern = []

for idx, row in df.iterrows():
    final_text = str(row['final_text']).lower().strip()
    if pd.isna(row['final_text']) or final_text == 'nan':
        continue
    
    # Check exact match
    if final_text in all_librispeech_texts:
        matches_found += 1
        text_type, lib_idx = all_librispeech_texts[final_text]
        
        # Check for pattern in misalignment
        if idx != lib_idx:
            misalignment_pattern.append((idx, lib_idx, lib_idx - idx))
        
        if matches_found <= 5:
            print(f"\nFOUND MATCH at CSV row {idx}:")
            print(f"  Final text: {row['final_text'][:60]}...")
            print(f"  Matches {text_type} text at LibriSpeech index {lib_idx}")
            print(f"  Offset: {lib_idx - idx}")

print(f"\n" + "="*60)
print(f"TOTAL EXACT MATCHES: {matches_found} out of {len(df)} ({100*matches_found/len(df):.1f}%)")

if matches_found > 900:
    print("\n🚨 HIGH MATCH RATE! The results ARE scrambled/misaligned.")
    print("The 'final_text' values are legitimate LibriSpeech transcriptions,")
    print("NOT model hallucinations. There's an indexing bug in the attack code.")
    
    # Analyze offset pattern
    if misalignment_pattern:
        offsets = [offset for _, _, offset in misalignment_pattern]
        from collections import Counter
        offset_counts = Counter(offsets)
        print(f"\nMost common offsets: {offset_counts.most_common(5)}")