#!/usr/bin/env python3
"""
Simple script to run the adversarial attack on a small test set.
This script is designed to work in the Docker container.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

def create_test_data(num_examples=10):
    """
    Create a test data file with specified number of examples from the LibriSpeech dataset.
    Format: audio_file_path, original_transcription, target_transcription
    These are the same 10 samples used in the original 2017 research.
    
    Args:
        num_examples: Number of examples to include (1-10)
    """
    all_examples = [
        "LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav,THE MORE SHE IS ENGAGED IN HER PROPER DUTIES THE LESS LEISURE WILL SHE HAVE FOR IT EVEN AS AN ACCOMPLISHMENT AND A RECREATION,OLD WILL IS A FINE FELLOW BUT POOR AND HELPLESS SINCE MISSUS ROGERS HAD HER ACCIDENT",
        "LibriSpeech/test-clean/2961/960/2961-960-0020.wav,AND HENCE WE FIND THE SAME SORT OF CLUMSINESS IN THE TIMAEUS OF PLATO WHICH CHARACTERIZES THE PHILOSOPHICAL POEM OF LUCRETIUS,THE WOOD FLAMED UP SPLENDIDLY UNDER THE LARGE BREWING COPPER AND IT SIGHED SO DEEPLY",
        "LibriSpeech/test-clean/2830/3980/2830-3980-0029.wav,ALTHOUGH THE BRETHREN WITH ME ARE NOT APOSTLES LIKE MYSELF YET THEY ARE ALL OF ONE MIND WITH ME THINK WRITE AND TEACH AS I DO,I WOULD FAIN KNOW IF I AM DESTINED FOR SO GLORIOUS A CAREER CRIED THE TREE REJOICING",
        "LibriSpeech/test-clean/2300/131720/2300-131720-0015.wav,HE OBTAINED THE DESIRED SPEED AND LOAD WITH A FRICTION BRAKE ALSO REGULATOR OF SPEED BUT WAITED FOR AN INDICATOR TO VERIFY IT,WE WILL GO OUT TOGETHER TO THE BOWER THERE IS A WAY DOWN TO THE COURT FROM MY WINDOW",
        "LibriSpeech/test-clean/8230/279154/8230-279154-0017.wav,THERE MAY BE A SPECIFIC FEELING WHICH COULD BE CALLED THE FEELING OF PASTNESS ESPECIALLY WHERE IMMEDIATE MEMORY IS CONCERNED,FINALLY THE ONE PARTY WENT OFF EXULTING AND THE OTHER WAS LEFT IN DESOLATION AND WOE",
        "LibriSpeech/test-clean/8224/274381/8224-274381-0007.wav,BY QUICK MARCHES THROUGH THESE INACCESSIBLE MOUNTAINS THAT GENERAL FREED HIMSELF FROM THE SUPERIOR FORCES OF THE COVENANTERS,SHE BLUSHED AND SMILED AND FUMBLED HIS CARD IN HER CONFUSION BEFORE SHE RAN UPSTAIRS",
        "LibriSpeech/test-clean/61/70968/61-70968-0049.wav,HAVE YOUR WILL CHILD IF THE BOY ALSO WILLS IT MONTFICHET ANSWERED FEELING TOO ILL TO OPPOSE ANYTHING VERY STRONGLY JUST THEN,THE MODERN ORGANIZATION OF INDUSTRY WORKS IN THE SAME DIRECTION ALSO BY ANOTHER LINE",
        "LibriSpeech/test-clean/61/70968/61-70968-0011.wav,HE GAVE WAY TO THE OTHERS VERY READILY AND RETREATED UNPERCEIVED BY THE SQUIRE AND MISTRESS FITZOOTH TO THE REAR OF THE TENT,ISN'T HE SPLENDID CRIED JASPER IN INTENSE PRIDE SWELLING UP FATHER KNEW HOW TO DO IT",
        "LibriSpeech/test-clean/5142/36377/5142-36377-0007.wav,A LITTLE CRACKED THAT IN THE POPULAR PHRASE WAS MY IMPRESSION OF THE STRANGER WHO NOW MADE HIS APPEARANCE IN THE SUPPER ROOM,HER REGARD SHIFTED TO THE GREEN STALKS AND LEAVES AGAIN AND SHE STARTED TO MOVE AWAY",
        "LibriSpeech/test-clean/5105/28241/5105-28241-0006.wav,THE LOG AND THE COMPASS THEREFORE WERE ABLE TO BE CALLED UPON TO DO THE WORK OF THE SEXTANT WHICH HAD BECOME UTTERLY USELESS,WHEN WE WERE OUT IN THE DARKNESS OF THE QUADRANGLE WE AGAIN LOOKED UP AT THE WINDOWS"
    ]
    
    # Validate num_examples
    num_examples = min(max(1, num_examples), len(all_examples))
    
    # Select the requested number of examples
    selected_examples = all_examples[:num_examples]
    test_data_content = "\n".join(selected_examples)
    
    with open("test_data.txt", "w") as f:
        f.write(test_data_content)
    print(f"Created test_data.txt with {num_examples} audio sample(s)")

def main():
    """Main function to run the attack."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run adversarial attack on ASR model")
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=10, 
        help="Number of examples to process (1-10, default: 10)"
    )
    parser.add_argument(
        "--num-iter-stage1",
        type=int,
        default=1000,
        help="Number of iterations for Stage 1 (default: 1000)"
    )
    parser.add_argument(
        "--num-iter-stage2",
        type=int,
        default=10,
        help="Number of iterations for Stage 2 (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-base",
        help="Model to use (default: openai/whisper-base)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu/cuda/mps/auto, default: auto detects best available)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Interval for logging predictions during optimization (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (log every iteration)"
    )
    parser.add_argument(
        "--lr-stage1",
        type=float,
        default=0.01,
        help="Learning rate for Stage 1 (default: 0.01)"
    )
    parser.add_argument(
        "--lr-stage2",
        type=float,
        default=0.001,
        help="Learning rate for Stage 2 (default: 0.001)"
    )
    parser.add_argument(
        "--initial-bound",
        type=float,
        default=0.05,
        help="Initial L-infinity bound for perturbations (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    # Validate num_examples
    if args.num_examples < 1 or args.num_examples > 10:
        print(f"Warning: num_examples must be between 1 and 10. Using default (10).")
        args.num_examples = 10
    
    print("=== Adversarial ASR Attack - Modern Implementation ===")
    print(f"Target: Process {args.num_examples} LibriSpeech audio sample(s) with {args.model}")
    
    # Create test data file with specified number of examples
    create_test_data(args.num_examples)
    
    # Check if we're running in the expected directory structure
    librispeech_path = Path("../adversarial_asr/LibriSpeech")
    if not librispeech_path.exists():
        print(f"Warning: LibriSpeech path not found at {librispeech_path}")
        print("Looking for audio files in current directory...")
        librispeech_path = Path("./LibriSpeech")
        if not librispeech_path.exists():
            print("Error: Cannot find LibriSpeech audio files")
            print("Please ensure audio files are available")
            return
    
    print(f"Using audio files from: {librispeech_path.absolute()}")
    
    # Initialize attack with parameters
    attack = AdversarialAttack(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        num_iter_stage1=args.num_iter_stage1,
        num_iter_stage2=args.num_iter_stage2,
        log_interval=args.log_interval,
        verbose=args.verbose,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        initial_bound=args.initial_bound
    )
    
    # Run attack
    try:
        attack.run_attack(
            data_file="test_data.txt",
            root_dir=str(librispeech_path.parent),
            output_dir="./output"
        )
        print("=== Attack completed successfully! ===")
        print("Check the ./output directory for adversarial audio files")
        
    except Exception as e:
        print(f"Error during attack: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
