#!/usr/bin/env python3
"""
Simple script to run the adversarial attack on a small test set.
This script is designed to work in the Docker container.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

def create_test_data():
    """
    Create a test data file with 10 examples from the LibriSpeech dataset.
    Format: audio_file_path, original_transcription, target_transcription
    These are the same 10 samples used in the original 2017 research.
    """
    test_data_content = """LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav,THE MORE SHE IS ENGAGED IN HER PROPER DUTIES THE LESS LEISURE WILL SHE HAVE FOR IT EVEN AS AN ACCOMPLISHMENT AND A RECREATION,OLD WILL IS A FINE FELLOW BUT POOR AND HELPLESS SINCE MISSUS ROGERS HAD HER ACCIDENT
LibriSpeech/test-clean/2961/960/2961-960-0020.wav,AND HENCE WE FIND THE SAME SORT OF CLUMSINESS IN THE TIMAEUS OF PLATO WHICH CHARACTERIZES THE PHILOSOPHICAL POEM OF LUCRETIUS,THE WOOD FLAMED UP SPLENDIDLY UNDER THE LARGE BREWING COPPER AND IT SIGHED SO DEEPLY
LibriSpeech/test-clean/2830/3980/2830-3980-0029.wav,ALTHOUGH THE BRETHREN WITH ME ARE NOT APOSTLES LIKE MYSELF YET THEY ARE ALL OF ONE MIND WITH ME THINK WRITE AND TEACH AS I DO,I WOULD FAIN KNOW IF I AM DESTINED FOR SO GLORIOUS A CAREER CRIED THE TREE REJOICING
LibriSpeech/test-clean/2300/131720/2300-131720-0015.wav,HE OBTAINED THE DESIRED SPEED AND LOAD WITH A FRICTION BRAKE ALSO REGULATOR OF SPEED BUT WAITED FOR AN INDICATOR TO VERIFY IT,WE WILL GO OUT TOGETHER TO THE BOWER THERE IS A WAY DOWN TO THE COURT FROM MY WINDOW
LibriSpeech/test-clean/8230/279154/8230-279154-0017.wav,THERE MAY BE A SPECIFIC FEELING WHICH COULD BE CALLED THE FEELING OF PASTNESS ESPECIALLY WHERE IMMEDIATE MEMORY IS CONCERNED,FINALLY THE ONE PARTY WENT OFF EXULTING AND THE OTHER WAS LEFT IN DESOLATION AND WOE
LibriSpeech/test-clean/8224/274381/8224-274381-0007.wav,BY QUICK MARCHES THROUGH THESE INACCESSIBLE MOUNTAINS THAT GENERAL FREED HIMSELF FROM THE SUPERIOR FORCES OF THE COVENANTERS,SHE BLUSHED AND SMILED AND FUMBLED HIS CARD IN HER CONFUSION BEFORE SHE RAN UPSTAIRS
LibriSpeech/test-clean/61/70968/61-70968-0049.wav,HAVE YOUR WILL CHILD IF THE BOY ALSO WILLS IT MONTFICHET ANSWERED FEELING TOO ILL TO OPPOSE ANYTHING VERY STRONGLY JUST THEN,THE MODERN ORGANIZATION OF INDUSTRY WORKS IN THE SAME DIRECTION ALSO BY ANOTHER LINE
LibriSpeech/test-clean/61/70968/61-70968-0011.wav,HE GAVE WAY TO THE OTHERS VERY READILY AND RETREATED UNPERCEIVED BY THE SQUIRE AND MISTRESS FITZOOTH TO THE REAR OF THE TENT,ISN'T HE SPLENDID CRIED JASPER IN INTENSE PRIDE SWELLING UP FATHER KNEW HOW TO DO IT
LibriSpeech/test-clean/5142/36377/5142-36377-0007.wav,A LITTLE CRACKED THAT IN THE POPULAR PHRASE WAS MY IMPRESSION OF THE STRANGER WHO NOW MADE HIS APPEARANCE IN THE SUPPER ROOM,HER REGARD SHIFTED TO THE GREEN STALKS AND LEAVES AGAIN AND SHE STARTED TO MOVE AWAY
LibriSpeech/test-clean/5105/28241/5105-28241-0006.wav,THE LOG AND THE COMPASS THEREFORE WERE ABLE TO BE CALLED UPON TO DO THE WORK OF THE SEXTANT WHICH HAD BECOME UTTERLY USELESS,WHEN WE WERE OUT IN THE DARKNESS OF THE QUADRANGLE WE AGAIN LOOKED UP AT THE WINDOWS"""
    
    with open("test_data.txt", "w") as f:
        f.write(test_data_content)
    print("Created test_data.txt with 10 audio samples")

def main():
    """Main function to run the attack."""
    print("=== Adversarial ASR Attack - Modern Implementation ===")
    print("Target: Process 10 LibriSpeech audio samples with Whisper")
    
    # Create test data file
    create_test_data()
    
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
    
    # Initialize attack with parameters for a real attack run
    attack = AdversarialAttack(
        model_name="openai/whisper-base",
        device='cpu',  # Force CPU for M-series Mac compatibility
        batch_size=1,  # Process one example at a time for clarity
        num_iter_stage1=1000, # Increased iterations for Stage 1
        num_iter_stage2=10    # Stage 2 is not yet implemented
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
