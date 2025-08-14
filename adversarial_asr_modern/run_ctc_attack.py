#!/usr/bin/env python3
"""
CLI script to run CTC-based adversarial attacks.
Provides easy access to the CTC attack functionality.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.ctc_attack import CTCAdversarialAttack


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Run CTC-based adversarial audio attack',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--data_file', default='../adversarial_asr/read_data.txt', 
                       help='Path to data file containing audio files and targets')
    parser.add_argument('--root_dir', default='../adversarial_asr/', 
                       help='Root directory containing audio files')
    parser.add_argument('--output_dir', default='./output_ctc', 
                       help='Output directory for adversarial examples')
    
    # Model configuration
    parser.add_argument('--model_size', default='base', 
                       choices=['base', 'large', 'large-lv60'],
                       help='CTC model size (base=95M, large=317M, large-lv60=317M+better)')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device to use (cpu/cuda/mps/auto, auto will detect best available)')
    
    # Processing configuration
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of examples to process from data file')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size for processing (memory vs speed tradeoff)')
    
    # Attack parameters
    parser.add_argument('--initial_bound', type=float, default=0.15,
                       help='Initial L-infinity bound for perturbations (fraction of signal range)')
    parser.add_argument('--lr_stage1', type=float, default=0.05,
                       help='Learning rate for stage 1 (effectiveness optimization)')
    parser.add_argument('--lr_stage2', type=float, default=0.005,
                       help='Learning rate for stage 2 (imperceptibility optimization)')
    parser.add_argument('--num_iter_stage1', type=int, default=1000,
                       help='Number of iterations for stage 1')
    parser.add_argument('--num_iter_stage2', type=int, default=4000,
                       help='Number of iterations for stage 2')
    
    # Feature flags
    parser.add_argument('--disable_stage2', action='store_true',
                       help='Disable stage 2 (psychoacoustic masking)')
    parser.add_argument('--test_model', action='store_true',
                       help='Test model functionality without running attacks')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("CTC ADVERSARIAL ATTACK")
    print("=" * 70)
    print(f"Model size:        {args.model_size}")
    print(f"Device:            {args.device}")
    print(f"Examples:          {args.num_examples}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Stage 2 enabled:   {not args.disable_stage2}")
    print(f"Data file:         {args.data_file}")
    print(f"Root dir:          {args.root_dir}")
    print(f"Output dir:        {args.output_dir}")
    print("=" * 70)
    
    if args.test_model:
        # Test model functionality
        print("Testing CTC model functionality...")
        from adversarial_asr_modern.ctc_audio_utils import test_ctc_model
        
        # Find a test audio file
        test_audio = None
        root_path = Path(args.root_dir)
        if root_path.exists():
            # Look for any .wav file in LibriSpeech structure
            for wav_file in root_path.rglob("*.wav"):
                test_audio = str(wav_file)
                break
        
        test_ctc_model(args.model_size, test_audio)
        return
    
    try:
        # Initialize attack
        attack = CTCAdversarialAttack(
            model_size=args.model_size,
            device=args.device,
            batch_size=args.batch_size,
            initial_bound=args.initial_bound,
            lr_stage1=args.lr_stage1,
            lr_stage2=args.lr_stage2,
            num_iter_stage1=args.num_iter_stage1,
            num_iter_stage2=args.num_iter_stage2
        )
        
        # Run attack
        attack.run_attack(
            data_file=args.data_file, 
            root_dir=args.root_dir, 
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            enable_stage2=not args.disable_stage2
        )
        
        print("\n" + "=" * 70)
        print("ATTACK COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {args.output_dir}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
