#!/usr/bin/env python3
"""Check if results are properly aligned with their audio files."""

import pandas as pd

df = pd.read_csv('results_full_20250815_012857.csv')

print("Checking alignment issues...")
print("="*60)

# Check for rows where audio_file looks wrong
issues = []
for idx, row in df.iterrows():
    audio_file = str(row['audio_file'])
    audio_name = str(row['audio_name'])
    
    # Check if audio_file contains a number instead of path
    try:
        float(audio_file)
        issues.append((idx, 'audio_file is a number', audio_file))
    except:
        pass
    
    # Check if audio_name is "stage1" or "stage2"
    if audio_name in ['stage1', 'stage2']:
        issues.append((idx, 'audio_name is stage marker', audio_name))
    
    # Check if the stem of audio_file matches audio_name
    if 'LibriSpeech' in audio_file and audio_name not in audio_file:
        issues.append((idx, 'audio_name mismatch', f'{audio_file} vs {audio_name}'))

print(f"Found {len(issues)} alignment issues")
print("\nFirst 10 issues:")
for issue in issues[:10]:
    print(f"  Row {issue[0]}: {issue[1]} - {issue[2]}")

# Check the data structure for pattern
print("\n" + "="*60)
print("Checking for column shift pattern...")

# Look at rows with problems
problem_rows = df[df['audio_name'].isin(['stage1', 'stage2', 'False', 'True'])]
print(f"\nRows with stage/bool values in audio_name: {len(problem_rows)}")

if len(problem_rows) > 0:
    print("\nExample problem rows:")
    cols = ['example_idx', 'audio_file', 'audio_name', 'stage', 'final_stage']
    print(problem_rows[cols].head(10).to_string())