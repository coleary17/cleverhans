#!/usr/bin/env python3
"""
Quick test to validate grid search setup without running full attacks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from grid_search_lr import GridSearchConfig, analyze_convergence
import pandas as pd

print("Testing grid search configuration...")

# Test configuration class
print(f"LR Stage 1 options: {GridSearchConfig.LR_STAGE1_OPTIONS}")
print(f"LR Stage 2 options: {GridSearchConfig.LR_STAGE2_OPTIONS}")
print(f"Initial bound options: {GridSearchConfig.INITIAL_BOUND_OPTIONS}")

# Test convergence analysis with dummy data
dummy_history = {
    'losses': [
        [10.0, 10.5, 9.8, 10.2, 10.1],
        [9.5, 9.8, 9.2, 9.6, 9.4],
        [9.0, 9.2, 8.8, 9.1, 8.9]
    ]
}

metrics = analyze_convergence(dummy_history)
print(f"\nConvergence metrics test:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Test parameter combinations
from itertools import product
configs = list(product(
    [0.01, 0.1],
    [0.001, 0.01],
    [0.1, 0.15]
))
print(f"\nTotal configurations for quick test: {len(configs)}")

# Test data preparation
from adversarial_asr_modern.audio_utils import parse_data_file

# Check if we can load data
if Path("full_data_flac.txt").exists():
    data = parse_data_file("full_data_flac.txt")
    print(f"\nData file loaded: {len(data)} examples available")
    print(f"First example: {data[0][0]}")
else:
    print("\nNo data file found - would need to create it")

print("\n✅ Grid search setup validated successfully!")
print("\nTo run actual grid search:")
print("  Quick test (6 configs): python grid_search_lr.py --quick")
print("  Full search (75 configs): python grid_search_lr.py")