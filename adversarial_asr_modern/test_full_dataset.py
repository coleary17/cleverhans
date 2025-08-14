#!/usr/bin/env python3
"""
Test script to verify full dataset loading and run attack on subset.
Perfect for Docker testing before committing to full 1000-file run.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack
from adversarial_asr_modern.audio_utils import parse_data_file


def test_full_dataset(num_examples=1, num_iterations=10):
    """
    Test the full dataset pipeline with limited examples.
    
    Args:
        num_examples: Number of examples to process (default 1)
        num_iterations: Number of attack iterations (default 10)
    """
    
    print("=" * 70)
    print(f"FULL DATASET TEST - {num_examples} example(s), {num_iterations} iterations")
    print("=" * 70)
    
    # Step 1: Check LibriSpeech is available
    librispeech_dir = Path("LibriSpeech/test-clean")
    if not librispeech_dir.exists():
        print("❌ LibriSpeech not found!")
        print("   This should be downloaded in Docker automatically")
        print("   Or run: ./download_librispeech.sh")
        return False
    
    flac_files = list(librispeech_dir.glob("**/*.flac"))
    print(f"✅ LibriSpeech found: {len(flac_files)} FLAC files")
    
    # Step 2: Convert full data file if needed
    full_data_file = Path("full_data_flac.txt")
    if not full_data_file.exists():
        print("\nConverting read_data_full.txt to FLAC paths...")
        
        # Check for original file
        original_file = Path("read_data_full.txt")
        if not original_file.exists():
            original_file = Path("../adversarial_asr/util/read_data_full.txt")
        
        if original_file.exists():
            import subprocess
            result = subprocess.run(
                ["python", "convert_full_data.py", "--format", "csv"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✅ Data file converted successfully")
            else:
                print(f"❌ Conversion failed: {result.stderr}")
                return False
        else:
            print("❌ Original data file not found")
            return False
    
    # Step 3: Load and verify data
    print("\nLoading data file...")
    data = parse_data_file(str(full_data_file))
    print(f"✅ Loaded {len(data)} entries from data file")
    
    # Show first few entries
    print("\nFirst 3 entries:")
    for i, (audio, orig, target) in enumerate(data[:3]):
        print(f"{i+1}. {audio}")
        print(f"   Original: {orig[:50]}...")
        print(f"   Target:   {target[:50]}...")
    
    # Step 4: Create subset for testing
    test_data_file = f"test_{num_examples}_examples.txt"
    with open(test_data_file, 'w') as f:
        for i in range(min(num_examples, len(data))):
            audio, orig, target = data[i]
            f.write(f"{audio},{orig},{target}\n")
    print(f"\n✅ Created test file: {test_data_file}")
    
    # Step 5: Run attack on subset
    print("\n" + "-" * 70)
    print(f"Running attack on {num_examples} example(s)...")
    print("-" * 70)
    
    try:
        attack = AdversarialAttack(
            model_name="openai/whisper-base",
            device="auto",  # Auto-detect GPU/CPU
            batch_size=min(num_examples, 5),
            initial_bound=0.15,
            lr_stage1=0.1,
            lr_stage2=0.01,
            num_iter_stage1=num_iterations,
            num_iter_stage2=max(2, num_iterations // 10),
            log_interval=max(1, num_iterations // 10),
            verbose=False,
            save_audio=False  # Don't save audio files
        )
        
        # Run attack with results saved to CSV
        results_file = f"test_results_{num_examples}ex_{num_iterations}iter.csv"
        attack.run_attack(
            data_file=test_data_file,
            root_dir=".",
            output_dir="./output_test",
            results_file=results_file
        )
        
        print("\n✅ Attack completed successfully!")
        
        # Check results file
        results_path = Path(results_file)
        if results_path.exists():
            print(f"\n✅ Results saved to: {results_file}")
            
            # Read and show summary
            import pandas as pd
            df = pd.read_csv(results_file)
            print(f"   Total rows: {len(df)}")
            print(f"   Successful attacks: {df['success'].sum() if 'success' in df.columns else 0}")
            
            # Show first few results
            if len(df) > 0:
                print("\nFirst result:")
                row = df.iloc[0]
                print(f"   Audio: {row.get('audio_name', 'N/A')}")
                print(f"   Success: {row.get('success', 'N/A')}")
                print(f"   Target: {row.get('target_text', 'N/A')[:50]}...")
                print(f"   Final: {row.get('final_text', 'N/A')[:50]}...")
        else:
            print(f"\n⚠️  Results file not created: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Attack failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test full dataset loading and attack"
    )
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=1,
        help="Number of examples to test (default: 1)"
    )
    parser.add_argument(
        "--num-iterations", 
        type=int, 
        default=10,
        help="Number of attack iterations (default: 10)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full attack (1000 examples, 1000 iterations) - WARNING: Takes hours!"
    )
    
    args = parser.parse_args()
    
    if args.full:
        print("WARNING: Full attack will take HOURS or DAYS!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
        success = test_full_dataset(1000, 1000)
    else:
        success = test_full_dataset(args.num_examples, args.num_iterations)
    
    print("\n" + "=" * 70)
    if success:
        print("TEST PASSED ✅")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run more examples: --num-examples 10")
        print("2. Run longer attack: --num-iterations 100")
        print("3. Run full dataset: --full (WARNING: Takes hours!)")
    else:
        print("TEST FAILED ❌")
        print("=" * 70)
        print("Check the errors above")


if __name__ == "__main__":
    main()