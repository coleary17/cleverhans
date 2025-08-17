#!/usr/bin/env python3
"""Check if final_text values are actually original texts from other samples."""

import pandas as pd

df = pd.read_csv('results_full_20250815_012857.csv')

print("Checking if final_texts are scrambled from other samples...")
print("="*60)

# Collect all original texts and final texts
all_originals = df['original_text'].tolist()
all_finals = df['final_text'].tolist()

# Check if final texts appear as originals elsewhere
scrambled_matches = []

for idx, final_text in enumerate(all_finals):
    if pd.isna(final_text) or final_text == '':
        continue
    
    # Check if this final text matches any original text (excluding its own)
    for j, orig in enumerate(all_originals):
        if j != idx and not pd.isna(orig):
            # Check for exact or partial match
            if final_text[:50].lower() in orig.lower() or orig[:50].lower() in final_text.lower():
                scrambled_matches.append({
                    'row': idx,
                    'final_text_preview': final_text[:80],
                    'matches_original_at': j,
                    'original_preview': orig[:80]
                })
                break

print(f"Found {len(scrambled_matches)} potential scrambled matches")

if scrambled_matches:
    print("\nFirst 10 scrambled matches:")
    for match in scrambled_matches[:10]:
        print(f"\nRow {match['row']}:")
        print(f"  Final text: {match['final_text_preview']}...")
        print(f"  Matches original at row {match['matches_original_at']}:")
        print(f"  Original: {match['original_preview']}...")

# Also check if there's a pattern - like offset by N rows
print("\n" + "="*60)
print("Checking for systematic offset pattern...")

offset_counts = {}
for match in scrambled_matches:
    offset = match['matches_original_at'] - match['row']
    offset_counts[offset] = offset_counts.get(offset, 0) + 1

if offset_counts:
    sorted_offsets = sorted(offset_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMost common offsets:")
    for offset, count in sorted_offsets[:5]:
        print(f"  Offset {offset}: {count} occurrences")