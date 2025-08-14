#!/bin/bash
# Setup script for UV environment
# Run this with: source setup_env.sh

echo "🚀 Setting up UV environment for Adversarial ASR Modern..."

# Check if UV is installed
if ! command -v uv &> /dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
    echo "📦 Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Add UV to PATH for this session
export PATH="$HOME/.local/bin:$PATH"

# Verify UV is available
if command -v uv &> /dev/null || [ -f "$HOME/.local/bin/uv" ]; then
    echo "✅ UV is available"
    
    # Show UV version
    if [ -f "$HOME/.local/bin/uv" ]; then
        $HOME/.local/bin/uv --version
    else
        uv --version
    fi
    
    # Sync dependencies
    echo "📚 Syncing dependencies..."
    if [ -f "$HOME/.local/bin/uv" ]; then
        $HOME/.local/bin/uv sync
    else
        uv sync
    fi
    
    echo ""
    echo "✨ Environment ready! Use these commands:"
    echo "  uv run python test_gradients.py    # Test gradient flow"
    echo "  uv run python run_attack.py         # Run Whisper attack"
    echo "  uv run python run_ctc_attack.py     # Run CTC attack"
    echo ""
    echo "💡 Tip: Always use 'uv run' before Python commands"
else
    echo "❌ Could not find or install UV"
    echo "Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi