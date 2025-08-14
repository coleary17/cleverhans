#!/usr/bin/env python3
"""
Aggressive attack parameters for Whisper model.
Uses higher learning rates and bounds to overcome Whisper's robustness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack
from run_attack import create_test_data


def main():
    """Run attack with aggressive parameters."""
    
    print("=" * 60)
    print("WHISPER ADVERSARIAL ATTACK - AGGRESSIVE PARAMETERS")
    print("=" * 60)
    print("Using higher learning rates to overcome model robustness")
    print("=" * 60)
    
    # Create test data for 1 example
    create_test_data(1)
    
    # Check audio path
    librispeech_path = Path("../adversarial_asr/LibriSpeech")
    if not librispeech_path.exists():
        librispeech_path = Path("./LibriSpeech")
        if not librispeech_path.exists():
            print("Error: Cannot find LibriSpeech audio files")
            return
    
    print(f"Using audio from: {librispeech_path.absolute()}\n")
    
    # Initialize attack with aggressive parameters
    attack = AdversarialAttack(
        model_name="openai/whisper-base",
        device="auto",
        batch_size=1,
        initial_bound=0.15,      # Higher bound (15% of signal range)
        lr_stage1=0.1,           # Much higher learning rate
        lr_stage2=0.01,          # Higher fine-tuning rate
        num_iter_stage1=3000,    # More iterations
        num_iter_stage2=50,      # More fine-tuning
        log_interval=100,        # Log every 100 iterations
        verbose=False
    )
    
    print("Attack parameters:")
    print(f"  - Initial bound: 0.15 (15% of signal)")
    print(f"  - Learning rate: 0.1 (100x original)")
    print(f"  - Iterations: 3000")
    print("-" * 60)
    
    # Run attack
    try:
        attack.run_attack(
            data_file="test_data.txt",
            root_dir=str(librispeech_path.parent),
            output_dir="./output"
        )
        print("\n" + "=" * 60)
        print("Attack complete! Check ./output/ for results")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()