#!/usr/bin/env python3
"""
Aggressive CTC attack parameters for wav2vec2 models.
Uses optimized settings for successful attacks.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.ctc_attack import CTCAdversarialAttack
from adversarial_asr_modern.ctc_audio_utils import parse_data_file


def main():
    """Run CTC attack with aggressive parameters."""
    
    print("=" * 70)
    print("CTC/WAV2VEC2 ADVERSARIAL ATTACK - AGGRESSIVE PARAMETERS")
    print("=" * 70)
    print("Optimized for successful attacks on robust CTC models")
    print("=" * 70)
    
    # Check for LibriSpeech data
    data_file = "../adversarial_asr/read_data.txt"
    root_dir = "../adversarial_asr/"
    
    if not Path(data_file).exists():
        print(f"Error: Data file not found at {data_file}")
        return
    
    if not Path(root_dir).exists():
        print(f"Error: Root directory not found at {root_dir}")
        return
    
    # Parse data to get first example
    data = parse_data_file(data_file)
    if not data:
        print("Error: No data found in file")
        return
    
    print(f"Processing: {data[0][0]}")
    print(f"Target: '{data[0][2]}'")
    print("-" * 70)
    
    # Initialize attack with aggressive parameters
    attack = CTCAdversarialAttack(
        model_size='base',
        device='auto',
        batch_size=1,
        initial_bound=0.2,       # 20% of signal range (aggressive)
        lr_stage1=0.1,           # High learning rate
        lr_stage2=0.01,          # High fine-tuning rate
        num_iter_stage1=2000,    # More iterations
        num_iter_stage2=100      # More fine-tuning
    )
    
    print("\nAttack parameters:")
    print(f"  - Model: wav2vec2-base-960h")
    print(f"  - Initial bound: 0.2 (20% of signal)")
    print(f"  - Learning rate: 0.1")
    print(f"  - Iterations: 2000")
    print(f"  - Using signed gradients (CTC optimization)")
    print("-" * 70)
    
    # Run attack on first example
    print("\nRunning attack...")
    
    try:
        attack.run_attack(
            data_file=data_file,
            root_dir=root_dir,
            output_dir="./output_ctc",
            num_examples=1,
            disable_stage2=False  # Enable psychoacoustic masking
        )
        
        print("\n" + "=" * 70)
        print("Attack complete! Check ./output_ctc/ for results")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error during attack: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()