#!/bin/bash
# Test Docker setup with full LibriSpeech dataset but only process 1 example
# This verifies the entire pipeline works before committing to a full run

set -e

echo "==========================================="
echo "Docker Test - Full Dataset, Single Example"
echo "==========================================="
echo ""
echo "This test will:"
echo "1. Build Docker image with LibriSpeech download"
echo "2. Convert full data file to FLAC paths"
echo "3. Run attack on just 1 example"
echo "4. Verify output is created"
echo ""

# Build the Docker image with dataset
echo "Building Docker image (this downloads LibriSpeech)..."
docker build -t adversarial-asr-test-full .

# Create output directory
mkdir -p output_test

# Run test with full dataset but only 1 example
echo ""
echo "Running test (1 example from 1000-file dataset)..."
docker run --rm \
  -v "$(pwd)/output_test:/app/output_test" \
  adversarial-asr-test-full \
  bash -c "
    echo '=== Inside Docker Container ==='
    echo ''
    
    # Show LibriSpeech is available
    echo 'Checking LibriSpeech dataset...'
    FLAC_COUNT=\$(find LibriSpeech/test-clean -name '*.flac' 2>/dev/null | wc -l)
    echo \"Found \${FLAC_COUNT} FLAC files in LibriSpeech\"
    
    # Convert the full data file
    echo ''
    echo 'Converting full data file to FLAC paths...'
    python convert_full_data.py --format csv
    
    # Check the converted file
    echo ''
    echo 'Data file created with:'
    wc -l full_data_flac.txt
    echo 'First 3 entries:'
    head -3 full_data_flac.txt | cut -d',' -f1
    
    # Create a test file with just the first example
    echo ''
    echo 'Creating single-example test file...'
    head -1 full_data_flac.txt > test_single.txt
    
    # Run attack on single example
    echo ''
    echo 'Running attack on 1 example (quick test)...'
    uv run python -c \"
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from adversarial_asr_modern.adversarial_attack import AdversarialAttack

# Show what we're attacking
with open('test_single.txt', 'r') as f:
    line = f.readline().strip()
    parts = line.split(',')
    print(f'Testing with:')
    print(f'  Audio: {parts[0]}')
    print(f'  Target: {parts[2][:50]}...')

# Run attack with minimal iterations
attack = AdversarialAttack(
    model_name='openai/whisper-base',
    device='cpu',  # CPU in Docker
    batch_size=1,
    initial_bound=0.15,
    lr_stage1=0.1,
    lr_stage2=0.01,
    num_iter_stage1=10,  # Just 10 iterations for test
    num_iter_stage2=2,
    log_interval=5,
    verbose=False
)

attack.run_attack(
    data_file='test_single.txt',
    root_dir='.',
    output_dir='./output_test'
)

print('')
print('Test attack completed!')
\"
    
    # Check output was created
    echo ''
    echo 'Checking output files...'
    ls -la output_test/*.wav 2>/dev/null || echo 'No output files created yet'
    
    echo ''
    echo '=== Test Complete ==='
  "

echo ""
echo "==========================================="
echo "Test Summary"
echo "==========================================="

# Check if output was created
if ls output_test/*.wav 1> /dev/null 2>&1; then
    echo "✅ SUCCESS: Attack produced output files"
    echo ""
    echo "Output files:"
    ls -la output_test/*.wav
else
    echo "⚠️  WARNING: No output files found"
    echo "Check the logs above for errors"
fi

echo ""
echo "Next steps:"
echo "1. If test passed, run full attack with:"
echo "   docker run --gpus all adversarial-asr-gpu"
echo "2. Or run more examples by modifying test_single.txt"
echo "==========================================="