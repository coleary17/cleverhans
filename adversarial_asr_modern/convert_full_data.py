#!/usr/bin/env python3
"""
Convert the original read_data_full.txt (WAV paths) to work with FLAC files.
Maintains the exact same ordering and transcriptions as the original research.
"""

import os
from pathlib import Path


def convert_full_data_to_flac(
    original_file="../adversarial_asr/util/read_data_full.txt",
    output_file="full_data_flac.txt",
    format="csv"  # or "original" for comma-separated lines
):
    """
    Convert the original full data file to use FLAC paths instead of WAV.
    
    Args:
        original_file: Path to original read_data_full.txt
        output_file: Output file with FLAC paths
        format: "csv" for line-by-line CSV or "original" for 3-line format
    """
    
    print("Converting full data file to FLAC paths...")
    
    # Read the original file
    with open(original_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) != 3:
        print(f"Error: Expected 3 lines, got {len(lines)}")
        return
    
    # Parse the three lines
    audio_files = lines[0].strip().split(',')
    original_texts = lines[1].strip().split(',')
    target_texts = lines[2].strip().split(',')
    
    # Check consistency
    num_files = len(audio_files)
    if len(original_texts) != num_files or len(target_texts) != num_files:
        print(f"Error: Inconsistent number of entries")
        print(f"  Audio files: {len(audio_files)}")
        print(f"  Original texts: {len(original_texts)}")
        print(f"  Target texts: {len(target_texts)}")
        return
    
    print(f"Found {num_files} entries in original file")
    
    # Convert WAV paths to FLAC paths
    flac_files = []
    missing_count = 0
    
    for wav_path in audio_files:
        # Convert .wav to .flac
        flac_path = wav_path.replace('.wav', '.flac')
        
        # Check if FLAC file exists (optional)
        if Path(flac_path).exists():
            print(f"✓ Found: {flac_path}")
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing
                print(f"✗ Missing: {flac_path}")
        
        flac_files.append(flac_path)
    
    if missing_count > 0:
        print(f"\nNote: {missing_count} FLAC files not found locally")
        print("They will be available after running ./download_librispeech.sh")
    
    # Write output file
    if format == "original":
        # Keep original 3-line format
        with open(output_file, 'w') as f:
            f.write(','.join(flac_files) + '\n')
            f.write(','.join(original_texts) + '\n')
            f.write(','.join(target_texts) + '\n')
        print(f"\nCreated {output_file} in original 3-line format")
        
    else:  # CSV format
        # Write as CSV (one entry per line)
        with open(output_file, 'w') as f:
            for i in range(num_files):
                f.write(f"{flac_files[i]},{original_texts[i]},{target_texts[i]}\n")
        print(f"\nCreated {output_file} with {num_files} entries in CSV format")
    
    # Show sample entries
    print("\nFirst 3 entries:")
    for i in range(min(3, num_files)):
        print(f"{i+1}. {flac_files[i]}")
        print(f"   Original: {original_texts[i][:60]}...")
        print(f"   Target:   {target_texts[i][:60]}...")


def create_attack_script():
    """Create a script to run the full 1000-file attack."""
    
    script_content = '''#!/usr/bin/env python3
"""
Run adversarial attack on full LibriSpeech dataset (1000 files).
This replicates the original 2017 research at scale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack


def main():
    print("=" * 70)
    print("FULL DATASET ADVERSARIAL ATTACK - 1000 FILES")
    print("=" * 70)
    
    # Check for data file
    data_file = "full_data_flac.txt"
    if not Path(data_file).exists():
        print(f"Error: {data_file} not found")
        print("Run: python convert_full_data.py first")
        return
    
    # Check for LibriSpeech
    if not Path("LibriSpeech/test-clean").exists():
        print("Error: LibriSpeech not found")
        print("Run: ./download_librispeech.sh first")
        return
    
    print("Starting attack on 1000 audio files...")
    print("This will take a LONG time!")
    
    # Initialize attack with production parameters
    attack = AdversarialAttack(
        model_name="openai/whisper-base",
        device="auto",  # Will use GPU if available
        batch_size=10,  # Process 10 at a time
        initial_bound=0.15,
        lr_stage1=0.1,
        lr_stage2=0.01,
        num_iter_stage1=1000,
        num_iter_stage2=100,
        log_interval=100,
        verbose=False
    )
    
    # Run attack
    attack.run_attack(
        data_file=data_file,
        root_dir=".",
        output_dir="./output_full"
    )
    
    print("\\nAttack complete!")
    print("Results saved to ./output_full/")


if __name__ == "__main__":
    main()
'''
    
    with open("run_full_attack.py", 'w') as f:
        f.write(script_content)
    
    os.chmod("run_full_attack.py", 0o755)
    print("\nCreated run_full_attack.py for running the full attack")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert full data file to FLAC")
    parser.add_argument("--format", choices=["csv", "original"], default="csv",
                       help="Output format: csv (one per line) or original (3 lines)")
    parser.add_argument("--create-script", action="store_true",
                       help="Also create run_full_attack.py script")
    
    args = parser.parse_args()
    
    # Convert the data file
    convert_full_data_to_flac(format=args.format)
    
    # Optionally create attack script
    if args.create_script:
        create_attack_script()