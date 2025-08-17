#!/usr/bin/env python3
"""Check for exact matches between final_text and original_text from other rows."""

import pandas as pd
from collections import Counter

df = pd.read_csv('results_full_20250815_012857.csv')

print("Checking for exact text scrambling...")
print("="*60)

# Create normalized versions for comparison (lowercase, stripped)
df['orig_normalized'] = df['original_text'].str.lower().str.strip()
df['final_normalized'] = df['final_text'].str.lower().str.strip()

exact_matches = []
partial_matches = []

for idx in range(len(df)):
    final = df.loc[idx, 'final_normalized']
    if pd.isna(final) or final == '':
        continue
    
    # Check against all OTHER rows' originals
    for j in range(len(df)):
        if j == idx:
            continue
        orig = df.loc[j, 'orig_normalized']
        if pd.isna(orig):
            continue
            
        # Check for exact match
        if final == orig:
            exact_matches.append((idx, j))
            print(f"EXACT MATCH: Row {idx} final = Row {j} original")
            print(f"  Text: {df.loc[idx, 'final_text'][:100]}...")
            break
        # Check for substantial partial match (first 30 chars)
        elif len(final) > 30 and len(orig) > 30 and final[:30] == orig[:30]:
            partial_matches.append((idx, j))

print(f"\nFound {len(exact_matches)} exact matches")
print(f"Found {len(partial_matches)} partial matches (first 30 chars)")

if not exact_matches and not partial_matches:
    print("\nNo scrambling detected - final texts appear to be genuinely different")
    print("\nSample of actual final texts (these are model hallucinations):")
    for i in range(5):
        if not pd.isna(df.loc[i, 'final_text']):
            print(f"\nRow {i}:")
            print(f"  Original: {df.loc[i, 'original_text'][:60]}...")
            print(f"  Target:   {df.loc[i, 'target_text'][:60]}...")  
            print(f"  Final:    {df.loc[i, 'final_text'][:60]}...")
else:
    print("\nOffset analysis:")
    offsets = [j - i for i, j in exact_matches + partial_matches]
    offset_counts = Counter(offsets)
    print(f"Most common offsets: {offset_counts.most_common(5)}")