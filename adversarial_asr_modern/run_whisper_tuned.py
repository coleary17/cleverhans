#!/usr/bin/env python3
"""
Tuned Whisper attack script with optimized parameters to avoid static/noise.
Use this for better quality adversarial examples.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack


def main():
    """Run Whisper attack with carefully tuned parameters."""
    
    parser = argparse.ArgumentParser(
        description="Run tuned Whisper adversarial attack (optimized to avoid static)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic options
    parser.add_argument("--num-examples", type=int, default=1,
                       help="Number of examples to process")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cpu/cuda/mps/auto)")
    
    # Tuned parameters for Whisper
    parser.add_argument("--lr-stage1", type=float, default=0.0005,
                       help="Stage 1 learning rate (tuned for Whisper)")
    parser.add_argument("--lr-stage2", type=float, default=0.00005,
                       help="Stage 2 learning rate (fine-tuning)")
    parser.add_argument("--initial-bound", type=float, default=0.015,
                       help="Max perturbation (1.5% of signal range)")
    parser.add_argument("--num-iter-stage1", type=int, default=2000,
                       help="Stage 1 iterations (more iterations, smaller steps)")
    parser.add_argument("--num-iter-stage2", type=int, default=20,
                       help="Stage 2 iterations")
    
    # Model options
    parser.add_argument("--model", type=str, default="openai/whisper-base",
                       help="Whisper model to use")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Logging interval")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WHISPER ADVERSARIAL ATTACK - TUNED PARAMETERS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Learning rates: Stage1={args.lr_stage1}, Stage2={args.lr_stage2}")
    print(f"Initial bound: {args.initial_bound} (max perturbation)")
    print(f"Iterations: Stage1={args.num_iter_stage1}, Stage2={args.num_iter_stage2}")
    print("=" * 60)
    
    # Create test data
    from run_attack import create_test_data
    create_test_data(args.num_examples)
    
    # Check audio path
    librispeech_path = Path("../adversarial_asr/LibriSpeech")
    if not librispeech_path.exists():
        librispeech_path = Path("./LibriSpeech")
        if not librispeech_path.exists():
            print("Error: Cannot find LibriSpeech audio files")
            return
    
    print(f"Using audio from: {librispeech_path.absolute()}")
    
    # Initialize attack with tuned parameters
    attack = AdversarialAttack(
        model_name=args.model,
        device=args.device,
        batch_size=1,  # Process one at a time for stability
        initial_bound=args.initial_bound,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        num_iter_stage1=args.num_iter_stage1,
        num_iter_stage2=args.num_iter_stage2,
        log_interval=args.log_interval,
        verbose=args.verbose
    )
    
    # Run attack
    try:
        attack.run_attack(
            data_file="test_data.txt",
            root_dir=str(librispeech_path.parent),
            output_dir="./output"
        )
        print("\n" + "=" * 60)
        print("SUCCESS! Check ./output/ for adversarial audio")
        print("The audio should sound clear without static/noise")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()