#!/bin/bash
# Run attack using system Python (for Lambda Labs and similar)

set -e

echo "============================================"
echo "Running with System Python"
echo "============================================"
echo ""

# Check what's available in system Python
echo "Checking system Python packages..."
python3 -c "
import sys
print(f'Python: {sys.version}')
print(f'Executable: {sys.executable}')
print()

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('❌ PyTorch not found')
    
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not found - installing...')
"

# Install missing packages to user directory
echo ""
echo "Installing any missing packages..."
pip3 install --user --quiet \
    transformers \
    openai-whisper \
    datasets \
    librosa \
    soundfile \
    audioread \
    pandas \
    absl-py

# Download LibriSpeech if needed
if [ ! -d "LibriSpeech/test-clean" ]; then
    echo ""
    echo "Downloading LibriSpeech dataset..."
    wget -q --show-progress http://www.openslr.org/resources/12/test-clean.tar.gz
    tar -xzf test-clean.tar.gz
    rm test-clean.tar.gz
    echo "✅ LibriSpeech downloaded"
fi

# Create simple test data
echo ""
echo "Creating test data..."
cat > test_attack_data.txt << 'DATA'
LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac,HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE,THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac,STUFF IT INTO YOU HIS BELLY COUNSELLED HIM,HELLO WORLD THIS IS A TEST
LibriSpeech/test-clean/1089/134686/1089-134686-0002.flac,AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS,ADVERSARIAL EXAMPLES ARE INTERESTING
DATA

# Run the attack
echo ""
echo "Running adversarial attack..."
echo "============================================"

python3 << 'PYTHON'
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

print("Importing modules...")
import torch
from adversarial_asr_modern.adversarial_attack import AdversarialAttack

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\nInitializing attack...")
attack = AdversarialAttack(
    model_name='openai/whisper-base',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=3,
    initial_bound=0.15,
    lr_stage1=0.05,
    lr_stage2=0.01,
    num_iter_stage1=30,
    num_iter_stage2=10,
    log_interval=10,
    verbose=True,
    save_audio=True,
    skip_stage2_on_failure=True
)

print("\nRunning attack on test examples...")
os.makedirs('output', exist_ok=True)
os.makedirs('results', exist_ok=True)

attack.run_attack(
    data_file='test_attack_data.txt',
    root_dir='.',
    output_dir='./output',
    results_file='./results/system_python_test.csv'
)

print("\n✅ Attack completed!")
print("Results saved to: ./results/system_python_test.csv")
print("Audio files saved to: ./output/")

# Show results
try:
    import pandas as pd
    df = pd.read_csv('./results/system_python_test.csv')
    print(f"\nResults summary:")
    print(f"  Total examples: {len(df)}")
    if 'stage1_success' in df.columns:
        print(f"  Stage 1 success: {df['stage1_success'].sum()}/{len(df)}")
    if 'stage2_success' in df.columns:
        print(f"  Stage 2 success: {df['stage2_success'].sum()}/{len(df)}")
except:
    pass
PYTHON

echo ""
echo "============================================"
echo "Test complete!"
echo "============================================"
