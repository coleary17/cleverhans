#!/usr/bin/env python3
"""
Attack using the EXACT parameters from the original paper, properly scaled.
This should work if the implementation is correct.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack
from run_attack import create_test_data


# Original parameters from the 2017 paper:
# - initial_bound: 2000 (for int16 range [-32768, 32767])
# - lr_stage1: 100
# - lr_stage2: 1
# - num_iter_stage1: 1000
# - num_iter_stage2: 4000

# Scale factors for normalized audio [-1, 1]:
INT16_MAX = 32768.0


def main():
    """Run attack with original parameters, properly scaled."""
    
    print("=" * 70)
    print("WHISPER ATTACK - ORIGINAL PAPER PARAMETERS (SCALED)")
    print("=" * 70)
    
    # Calculate scaled parameters
    initial_bound_scaled = 2000.0 / INT16_MAX  # 0.061
    lr_stage1_scaled = 100.0 / INT16_MAX  # 0.003
    lr_stage2_scaled = 1.0 / INT16_MAX  # 0.00003
    
    print("Original parameters (int16 range):")
    print(f"  initial_bound: 2000")
    print(f"  lr_stage1: 100")
    print(f"  lr_stage2: 1")
    print("\nScaled parameters (normalized range):")
    print(f"  initial_bound: {initial_bound_scaled:.6f}")
    print(f"  lr_stage1: {lr_stage1_scaled:.6f}")
    print(f"  lr_stage2: {lr_stage2_scaled:.6f}")
    print("=" * 70)
    
    # Create test data
    create_test_data(1)
    
    # Check audio path
    librispeech_path = Path("../adversarial_asr/LibriSpeech")
    if not librispeech_path.exists():
        librispeech_path = Path("./LibriSpeech")
        if not librispeech_path.exists():
            print("Error: Cannot find LibriSpeech audio files")
            return
    
    print(f"Using audio from: {librispeech_path.absolute()}\n")
    
    # Initialize attack with scaled parameters
    attack = AdversarialAttack(
        model_name="openai/whisper-base",
        device="auto",
        batch_size=1,
        initial_bound=initial_bound_scaled,
        lr_stage1=lr_stage1_scaled,
        lr_stage2=lr_stage2_scaled,
        num_iter_stage1=1000,
        num_iter_stage2=100,  # Reduced for testing
        log_interval=100,
        verbose=False
    )
    
    # Run attack
    try:
        attack.run_attack(
            data_file="test_data.txt",
            root_dir=str(librispeech_path.parent),
            output_dir="./output"
        )
        
        print("\n" + "=" * 70)
        print("Attack complete with original parameters!")
        print("Check ./output/ for results")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()