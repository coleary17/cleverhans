# UV Package Manager Guide

UV is a fast, modern Python package manager that replaces pip, virtualenv, and other tools. This project is configured to use UV for all dependency management.

## Installation

UV is already installed system-wide. If you need to reinstall:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## Essential UV Commands

### Always Start With This
```bash
# Add UV to your current shell session
source $HOME/.local/bin/env
```

### Project Setup (First Time)
```bash
# Clone the repository and navigate to it
cd adversarial_asr_modern

# Install all dependencies (creates .venv automatically)
uv sync
```

### Daily Development Commands

#### Running Python Scripts
```bash
# Always use 'uv run' to ensure the right environment
uv run python test_gradients.py
uv run python run_attack.py
uv run python run_ctc_attack.py
```

#### Adding New Dependencies
```bash
# Add a new package
uv add package-name

# Add a specific version
uv add package-name==1.2.3

# Add development dependencies
uv add --dev pytest black ruff
```

#### Removing Dependencies
```bash
uv remove package-name
```

#### Updating Dependencies
```bash
# Update all packages to latest compatible versions
uv sync --upgrade

# Update a specific package
uv add package-name --upgrade
```

#### Listing Installed Packages
```bash
uv pip list
```

## Project Structure

- `pyproject.toml` - Defines project metadata and dependencies
- `uv.lock` - Locks exact versions for reproducibility (auto-generated)
- `.venv/` - Virtual environment (auto-created by UV)
- `requirements.txt` - Legacy file (not used with UV)

## Common Workflows

### 1. Running Tests
```bash
uv run python test_installation.py
uv run python test_gradients.py
uv run python test_audio_outputs.py
uv run python test_ctc_model.py
```

### 2. Running Attacks
```bash
# Whisper attack
uv run python run_attack.py

# CTC (Wav2Vec2) attack
uv run python run_ctc_attack.py
```

### 3. Debugging
```bash
uv run python debug_whisper.py
uv run python diagnose_distortion.py
```

### 4. Interactive Python
```bash
# Start Python REPL with all dependencies available
uv run python

# Or with IPython if installed
uv add ipython
uv run ipython
```

## Docker Integration

When building Docker images, UV is used inside the container:

```dockerfile
FROM python:3.9-slim

# Install UV in container
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies with UV
RUN uv sync --frozen --no-dev

# Run with UV
CMD ["uv", "run", "python", "run_attack.py"]
```

## Troubleshooting

### Command 'uv' not found
```bash
# Add UV to your PATH
source $HOME/.local/bin/env

# Or add permanently to your shell config
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Dependencies not found when running scripts
```bash
# Always use 'uv run' instead of plain 'python'
uv run python script.py  # ✅ Correct
python script.py         # ❌ Won't have access to dependencies
```

### Virtual environment issues
```bash
# Remove and recreate the virtual environment
rm -rf .venv
uv sync
```

### Lock file conflicts
```bash
# Regenerate the lock file
rm uv.lock
uv sync
```

## Best Practices

1. **Always use `uv run`** - This ensures the correct environment is active
2. **Commit `uv.lock`** - This ensures reproducible builds
3. **Don't mix pip and uv** - Use UV for all package operations
4. **Keep pyproject.toml clean** - Only specify direct dependencies
5. **Use `uv sync` after pulling** - This ensures you have the right dependencies

## Migration from pip/requirements.txt

This project has already been migrated from pip to UV. The old `requirements.txt` is kept for reference but is not used. All dependencies are now managed through `pyproject.toml`.

## Quick Reference Card

```bash
# Setup
source $HOME/.local/bin/env  # Enable UV in current shell
uv sync                       # Install/update all dependencies

# Run
uv run python script.py       # Run any Python script
uv run pytest                 # Run tests

# Manage
uv add package               # Add new dependency
uv remove package            # Remove dependency
uv pip list                  # List installed packages
uv sync --upgrade            # Update all packages

# Clean
rm -rf .venv && uv sync      # Fresh install
```

## Environment Variables

UV respects standard Python environment variables:
- `VIRTUAL_ENV` - Path to virtual environment
- `UV_CACHE_DIR` - Cache directory for UV (default: `~/.cache/uv`)
- `UV_NO_CACHE` - Disable caching