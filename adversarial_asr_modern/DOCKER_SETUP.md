# Docker Setup and Audio Mounting Guide

## Audio Samples Source

The adversarial attack uses **10 specific audio samples from the LibriSpeech test-clean dataset** that are already included in the original `adversarial_asr` project directory. These are the exact same samples used in the original 2017 research paper.

### Audio File Locations
The audio files are located in:
```
adversarial_asr/LibriSpeech/test-clean/
├── 61/70968/61-70968-0011.wav
├── 61/70968/61-70968-0049.wav
├── 2300/131720/2300-131720-0015.wav
├── 2830/3980/2830-3980-0029.wav
├── 2961/960/2961-960-0020.wav
├── 3575/170457/3575-170457-0013.wav
├── 5105/28241/5105-28241-0006.wav
├── 5142/36377/5142-36377-0007.wav
├── 8224/274381/8224-274381-0007.wav
└── 8230/279154/8230-279154-0017.wav
```

## Docker Volume Mounting

### Step-by-Step Instructions

1. **Ensure you have both directories:**
   ```
   /your/path/to/cleverhans/
   ├── adversarial_asr/           # Original project with audio files
   │   └── LibriSpeech/
   │       └── test-clean/        # Contains the 10 .wav files
   └── adversarial_asr_modern/    # New modernized project
       ├── Dockerfile
       ├── run_attack.py
       └── ...
   ```

2. **Build the Docker image:**
   ```bash
   cd adversarial_asr_modern
   docker build -t adversarial-asr-modern .
   ```

3. **Run with volume mounts:**
   ```bash
   # From the adversarial_asr_modern directory
   docker run \
     -v "$(pwd)/../adversarial_asr/LibriSpeech:/app/LibriSpeech" \
     -v "$(pwd)/output:/app/output" \
     adversarial-asr-modern
   ```

### Mount Explanation

The Docker run command uses two volume mounts:

1. **Audio Input Mount:**
   - **Host Path:** `$(pwd)/../adversarial_asr/LibriSpeech`
   - **Container Path:** `/app/LibriSpeech`
   - **Purpose:** Makes the original LibriSpeech audio files available inside the container

2. **Output Mount:**
   - **Host Path:** `$(pwd)/output`
   - **Container Path:** `/app/output`
   - **Purpose:** Saves generated adversarial audio files back to your host machine

### Data Format

The attack script processes a CSV-like format where each line contains:
```
audio_file_path, original_transcription, target_transcription
```

For example:
```
LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav,THE MORE SHE IS ENGAGED IN HER PROPER DUTIES...,OLD WILL IS A FINE FELLOW BUT POOR...
```

### Expected Output

After running successfully, you'll find adversarial audio files in the `output/` directory:
```
output/
├── 3575-170457-0013_stage1.wav
├── 2961-960-0020_stage1.wav
├── 2830-3980-0029_stage1.wav
├── ...
└── (10 adversarial audio files total)
```

## Troubleshooting

### Common Issues

1. **"LibriSpeech not found" error:**
   - Ensure the original `adversarial_asr` directory exists alongside `adversarial_asr_modern`
   - Check that the LibriSpeech audio files are present in the expected locations

2. **Permission errors:**
   - Ensure Docker has access to read the audio files and write to the output directory
   - On some systems, you may need to adjust file permissions

3. **Platform issues on M-series Mac:**
   - The Dockerfile specifically uses `--platform=linux/arm64` for M-series compatibility
   - If you encounter issues, try building with `--platform=linux/amd64` flag

### Alternative Local Execution

If Docker mounting becomes complex, you can also run locally:

```bash
# Ensure uv is installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies and run
cd adversarial_asr_modern
uv sync
uv run python run_attack.py
```

This approach automatically detects the relative paths to the audio files without needing explicit mounting.
