#!/bin/bash
# Download LibriSpeech test-clean dataset for adversarial ASR experiments

set -e

echo "============================================"
echo "LibriSpeech Dataset Downloader"
echo "============================================"
echo ""

# Check if dataset already exists
if [ -d "LibriSpeech/test-clean" ]; then
    FLAC_COUNT=$(find LibriSpeech/test-clean -name '*.flac' 2>/dev/null | wc -l)
    echo "✅ LibriSpeech test-clean already exists"
    echo "   Found ${FLAC_COUNT} FLAC files"
    echo ""
    read -p "Re-download? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing dataset."
        exit 0
    fi
    echo "Removing existing dataset..."
    rm -rf LibriSpeech/
fi

echo "Downloading LibriSpeech test-clean dataset..."
echo "Size: ~350MB"
echo ""

# Download function
download_dataset() {
    local url="http://www.openslr.org/resources/12/test-clean.tar.gz"
    
    if command -v wget &> /dev/null; then
        echo "Using wget..."
        wget -q --show-progress "$url" -O test-clean.tar.gz
    elif command -v curl &> /dev/null; then
        echo "Using curl..."
        curl -L --progress-bar "$url" -o test-clean.tar.gz
    else
        echo "❌ Error: Neither wget nor curl found"
        echo "Please install one:"
        echo "  Ubuntu/Debian: sudo apt-get install wget"
        echo "  macOS: brew install wget"
        return 1
    fi
}

# Download the dataset
if download_dataset; then
    echo ""
    echo "Download complete. Extracting..."
    
    # Extract
    tar -xzf test-clean.tar.gz
    
    # Verify
    if [ -d "LibriSpeech/test-clean" ]; then
        FLAC_COUNT=$(find LibriSpeech/test-clean -name '*.flac' 2>/dev/null | wc -l)
        echo ""
        echo "✅ SUCCESS: LibriSpeech test-clean installed"
        echo "   Location: ./LibriSpeech/test-clean/"
        echo "   Files: ${FLAC_COUNT} FLAC audio files"
        
        # Show sample structure
        echo ""
        echo "Sample directory structure:"
        find LibriSpeech/test-clean -type d | head -5 | sed 's/^/  /'
        
        # Cleanup
        rm -f test-clean.tar.gz
        echo ""
        echo "Archive removed to save space."
    else
        echo "❌ Error: Failed to extract dataset"
        exit 1
    fi
else
    echo "❌ Download failed"
    exit 1
fi

echo ""
echo "============================================"
echo "Dataset ready for adversarial ASR attacks!"
echo "============================================"