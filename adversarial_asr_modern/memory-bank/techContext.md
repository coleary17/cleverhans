# Technology Context

## Technology Stack

### Core Technologies
- **Python 3.11**: Modern Python with improved performance and type hints
- **PyTorch 2.0+**: Deep learning framework with improved compilation and performance
- **Transformers 4.35+**: Hugging Face library for Whisper model access
- **librosa 0.10+**: Audio processing and feature extraction
- **numpy 1.24+**: Numerical computations and array operations

### Package Management
- **UV 0.1.18+**: Ultra-fast Python package installer and resolver
- **pyproject.toml**: Modern Python project configuration
- **uv.lock**: Reproducible dependency resolution with 77 packages

### Development Environment
- **Docker**: Containerization for reproducible deployments
- **Multi-architecture**: ARM64 (Apple Silicon) + x86_64 (Intel/AMD) support
- **Base Image**: `python:3.11-slim` for minimal footprint

## Development Setup

### Local Development (macOS ARM64)
```bash
# Prerequisites
brew install uv
git clone <repository>
cd adversarial_asr_modern

# Setup
uv sync                           # Install all dependencies
uv run python run_attack.py      # Run attack
uv run python debug_whisper.py   # Debug Whisper integration
```

### Docker Development
```bash
# Build multi-architecture image
docker build -t adversarial-asr-modern .

# Run with audio files mounted
docker run -v "$(pwd)/../adversarial_asr/LibriSpeech:/app/LibriSpeech" \
           -v "$(pwd)/output:/app/output" \
           adversarial-asr-modern
```

### Cloud GPU Deployment
```dockerfile
# Modify Dockerfile for GPU support
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# Rest of setup remains the same
```

## Technical Constraints

### Hardware Constraints
- **Memory**: Minimum 8GB RAM for Whisper base model
- **Storage**: ~2GB for models and dependencies
- **Compute**: CPU-only functional but slow (10-100x slower than GPU)

### Software Constraints
- **Python Version**: Requires Python 3.9+ for modern type hints
- **PyTorch Version**: Must be 2.0+ for optimal performance
- **CUDA Compatibility**: If using GPU, requires CUDA 11.8+ or 12.x

### Model Constraints
- **Whisper Model Size**: Using 'base' model (244MB) for balance of speed/quality
- **Audio Format**: Requires 16kHz mono audio for Whisper compatibility
- **Sequence Length**: Limited by Whisper's 30-second maximum input length

## Dependencies Analysis

### Production Dependencies (77 packages)
**Core ML Stack**:
```
torch==2.1.0                    # Deep learning framework
torchaudio==2.1.0               # Audio processing for PyTorch
transformers==4.35.2           # Hugging Face model access
librosa==0.10.1                 # Audio analysis and processing
soundfile==0.12.1              # Audio file I/O
```

**Scientific Computing**:
```
numpy==1.24.4                  # Numerical arrays and computations
scipy==1.11.3                  # Scientific computing utilities
```

**Development Tools**:
```
tqdm==4.66.1                   # Progress bars for long operations
```

### Key Dependency Relationships
- **transformers** → **torch**: Model implementations built on PyTorch
- **librosa** → **soundfile**: Audio loading and processing pipeline
- **torchaudio** → **torch**: Audio-specific PyTorch operations
- All packages pinned to specific versions for reproducibility

### Dependency Management Strategy
- **UV for speed**: 10-100x faster than pip for dependency resolution
- **Lock file**: `uv.lock` ensures identical installs across environments
- **Version pinning**: Exact versions to prevent compatibility issues
- **Minimal dependencies**: Only essential packages to reduce attack surface

## Tool Usage Patterns

### UV Package Manager
```bash
uv add <package>               # Add new dependency
uv remove <package>            # Remove dependency
uv sync                        # Install/sync all dependencies
uv run <command>               # Run command in virtual environment
uv lock                        # Update lock file
```

### PyTorch Device Management
```python
# Automatic device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tensor operations
tensor = torch.tensor(data).to(device)
model = model.to(device)
```

### Whisper Integration Pattern
```python
# Load model and processor together
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Process audio
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
outputs = model.generate(inputs["input_features"], language="en")
```

## Performance Optimization

### Memory Management
- **Model Loading**: Load once, reuse across multiple audio files
- **Batch Processing**: Process multiple samples together when possible
- **Gradient Accumulation**: For large batches that don't fit in memory
- **CPU Offloading**: Move intermediate results to CPU when GPU memory limited

### Computation Optimization
- **Device Placement**: Keep tensors on GPU throughout computation
- **Mixed Precision**: Use float16 for Whisper inference (future enhancement)
- **Compilation**: PyTorch 2.0 compilation for faster execution
- **Vectorization**: Process audio samples in batches rather than loops

### I/O Optimization
- **Lazy Loading**: Load audio files only when needed
- **Caching**: Cache computed features when processing multiple iterations
- **Async I/O**: Overlap file I/O with computation (future enhancement)

## Security Considerations

### Container Security
- **Minimal Base Image**: python:3.11-slim reduces attack surface
- **No Root User**: Run container as non-root user
- **Read-Only Filesystem**: Mount input/output directories with appropriate permissions

### Dependency Security
- **Pinned Versions**: All dependencies locked to specific versions
- **Vulnerability Scanning**: Regular dependency updates for security patches
- **Minimal Dependencies**: Only essential packages to reduce exposure

### Data Security
- **Local Processing**: No external API calls or data transmission
- **Temporary Files**: Clean up intermediate files after processing
- **Input Validation**: Validate audio files before processing

## Integration Patterns

### File System Integration
```python
# Standard paths
INPUT_DIR = "../adversarial_asr/LibriSpeech"
OUTPUT_DIR = "./output"
MODEL_CACHE = "~/.cache/huggingface"
```

### Docker Volume Mounting
```bash
# Mount necessary directories
-v "$(pwd)/../adversarial_asr/LibriSpeech:/app/LibriSpeech"  # Input audio
-v "$(pwd)/output:/app/output"                               # Output results
```

### Monitoring and Debugging
- **debug_whisper.py**: Test Whisper transcription in isolation
- **verbose logging**: Detailed output during attack execution  
- **intermediate outputs**: Loss values, gradients, audio statistics
- **error handling**: Graceful degradation with informative messages
