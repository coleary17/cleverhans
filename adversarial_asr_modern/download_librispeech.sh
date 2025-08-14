#!/bin/bash
# Download LibriSpeech test-clean dataset for adversarial attacks
# This dataset contains 2620 FLAC audio files from 40 speakers

set -e

echo "=========================================="
echo "LibriSpeech Test-Clean Dataset Downloader"
echo "=========================================="

# Configuration
DATASET_URL="https://www.openslr.org/resources/12/test-clean.tar.gz"
DATASET_DIR="LibriSpeech"
DATASET_SIZE="346MB"

# Check if dataset already exists
if [ -d "${DATASET_DIR}/test-clean" ]; then
    echo "Dataset already exists at ${DATASET_DIR}/test-clean"
    echo "To re-download, remove the existing directory first"
    exit 0
fi

echo ""
echo "This will download the LibriSpeech test-clean dataset"
echo "Dataset size: ${DATASET_SIZE} (compressed)"
echo "Contains: 2620 FLAC audio files"
echo ""

# Create directory if it doesn't exist
mkdir -p ${DATASET_DIR}

# Download dataset
echo "Downloading LibriSpeech test-clean..."
wget -q --show-progress ${DATASET_URL} -O test-clean.tar.gz

# Extract dataset
echo ""
echo "Extracting dataset..."
tar -xzf test-clean.tar.gz

# Clean up
rm test-clean.tar.gz

# Verify extraction
FLAC_COUNT=$(find ${DATASET_DIR}/test-clean -name "*.flac" | wc -l)
echo ""
echo "=========================================="
echo "Download complete!"
echo "Location: ${DATASET_DIR}/test-clean/"
echo "Files found: ${FLAC_COUNT} FLAC files"
echo "=========================================="

# Show sample structure
echo ""
echo "Sample directory structure:"
find ${DATASET_DIR}/test-clean -name "*.flac" | head -3 | while read -r file; do
    echo "  $file"
done
echo "  ..."