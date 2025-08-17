#!/usr/bin/env python3
"""Test script to verify iteration history tracking."""

import sys
sys.path.append('src')

from adversarial_asr_modern.adversarial_attack import AdversarialAttack
import json
from pathlib import Path

# Create a small test data file
test_data = """LibriSpeech/test-clean/3575/170457/3575-170457-0013.flac,THE MORE SHE IS ENGAGED IN HER PROPER DUTIES THE LESS LEISURE WILL SHE HAVE FOR IT EVEN AS AN ACCOMPLISHMENT AND A RECREATION,OLD WILL IS A FINE FELLOW BUT POOR AND HELPLESS SINCE MISSUS ROGERS HAD HER ACCIDENT
LibriSpeech/test-clean/2961/960/2961-960-0020.flac,AND HENCE WE FIND THE SAME SORT OF CLUMSINESS IN THE TIMAEUS OF PLATO WHICH CHARACTERIZES THE PHILOSOPHICAL POEM OF LUCRETIUS,THE WOOD FLAMED UP SPLENDIDLY UNDER THE LARGE BREWING COPPER AND IT SIGHED SO DEEPLY"""

with open('test_history_data.txt', 'w') as f:
    f.write(test_data)

# Run attack with history tracking
attack = AdversarialAttack(
    model_name="openai/whisper-base",
    device='auto',
    batch_size=2,
    num_iter_stage1=200,  # Run only 200 iterations for testing
    log_interval=50,
    verbose=False,
    save_audio=False,
    skip_stage2_on_failure=True
)

print("Running attack with history tracking...")
print("History will be saved every 100 iterations")
print("-" * 60)

attack.run_attack(
    data_file='test_history_data.txt',
    root_dir='../adversarial_asr/',
    output_dir='./test_output',
    results_file='test_history_results.csv'
)

# Check if history file was created
history_file = Path('test_history_results_history.json')
if history_file.exists():
    print(f"\n✓ History file created: {history_file}")
    
    # Load and analyze history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print(f"History checkpoints: {history['iterations']}")
    print(f"Number of checkpoints: {len(history['iterations'])}")
    
    # Show evolution of first example
    if history['transcriptions']:
        print("\nEvolution of Example 0:")
        for i, checkpoint in enumerate(history['iterations']):
            trans = history['transcriptions'][i][0]  # First example
            loss = history['losses'][i][0]  # First example loss
            pert_stats = history['perturbation_stats'][i][0]
            
            print(f"\nIteration {checkpoint}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Max perturbation: {pert_stats['max_perturbation']:.4f}")
            print(f"  Target: {trans['target'][:50]}...")
            print(f"  Prediction: {trans['prediction'][:50]}...")
            print(f"  Success: {trans['success']}")
else:
    print(f"\n✗ History file not found")

print("\nTest complete!")