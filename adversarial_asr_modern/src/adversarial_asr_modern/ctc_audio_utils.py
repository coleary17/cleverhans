"""
CTC model utilities for adversarial audio attacks.
Provides wav2vec2 CTC model wrapper with proper gradient flow support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)

# Model size configurations
MODEL_CONFIGS = {
    'base': 'facebook/wav2vec2-base-960h',
    'large': 'facebook/wav2vec2-large-960h', 
    'large-lv60': 'facebook/wav2vec2-large-960h-lv60-self'
}


class CTCASRModel:
    """
    Wav2Vec2 CTC model wrapper optimized for adversarial attacks.
    Ensures proper gradient flow from audio input to loss computation.
    """
    
    def __init__(self, model_size: str = 'base', device: str = 'auto'):
        """
        Initialize CTC ASR model.
        
        Args:
            model_size: Model size ('base', 'large', 'large-lv60')
            device: Device to use ('cpu', 'cuda', or 'auto' for auto-detection)
        """
        # Auto-detect best available device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        elif device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, falling back to CPU")
            self.device = torch.device('cpu')
        elif device == 'mps':
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
            else:
                print(f"Warning: MPS requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.model_size = model_size
        
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Model size must be one of {list(MODEL_CONFIGS.keys())}")
        
        model_name = MODEL_CONFIGS[model_size]
        print(f"Loading {model_name} on {self.device}")
        
        # Load model components
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        # Set model to eval mode but allow gradients through input
        # We don't need to train the model, just need gradients w.r.t. input
        self.model.eval()
        
        # Don't modify model parameter gradients - keep them as they are
        # We only need gradients w.r.t. the input audio
        
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio without gradients (for evaluation).
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        with torch.no_grad():
            # Ensure audio is normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Resample if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Process audio
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(self.device)
            
            # Get model output
            with torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.lower().strip()
    
    def compute_loss_differentiable(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Compute CTC loss with full gradient support.

        Args:
            audio_tensor: Audio tensor with requires_grad=True
            target_text: Target transcription text
            
        Returns:
            Differentiable loss tensor
        """
        # Ensure proper shape [batch_size, sequence_length]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Normalize to [-1, 1] (wav2vec2 requirement)
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
        
        # Get target tokens
        with torch.no_grad():
            labels = self.processor.tokenizer(target_text, return_tensors="pt").input_ids
            labels = labels.to(self.device)
        
        # Forward pass - wav2vec2 handles feature extraction internally
        outputs = self.model(input_values=audio_tensor, labels=labels)
        return outputs.loss
    
    def compute_attack_loss(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Compute loss for adversarial attack (minimize to achieve target).
        
        Args:
            audio_tensor: Audio tensor with gradients
            target_text: Target text to achieve
            
        Returns:
            Loss to minimize (negative CTC loss)
        """
        # Get logits with gradients
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Normalize if needed
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
        
        # Forward pass
        outputs = self.model(input_values=audio_tensor)
        logits = outputs.logits
        
        # Manual CTC loss for better control
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Target preparation
        with torch.no_grad():
            targets = self.processor.tokenizer(target_text, return_tensors="pt").input_ids
            targets = targets.to(self.device)
        
        # Compute CTC loss
        input_lengths = torch.full((1,), logits.shape[1], device=self.device)
        target_lengths = torch.full((1,), targets.shape[1], device=self.device)
        
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
            blank=self.processor.tokenizer.pad_token_id,
            zero_infinity=True
        )
        
        return -loss  # Negative for attack
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_size': self.model_size,
            'model_name': MODEL_CONFIGS[self.model_size],
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'vocab_size': self.processor.tokenizer.vocab_size
        }


def load_audio_file(filepath: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        filepath: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ensure float32 and proper range
        audio = audio.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        return audio, target_sr
        
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {filepath}: {e}")


def save_audio_file(audio: np.ndarray, filepath: str, sample_rate: int = 16000):
    """
    Save audio array to file.
    
    Args:
        audio: Audio array
        filepath: Output file path
        sample_rate: Sample rate
    """
    try:
        # Ensure audio is in proper format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure proper range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Create output directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        sf.write(filepath, audio, sample_rate)
        
    except Exception as e:
        raise RuntimeError(f"Error saving audio file {filepath}: {e}")


def audio_to_tensor(audio: np.ndarray, device: str = 'cpu', requires_grad: bool = True) -> torch.Tensor:
    """
    Convert numpy audio array to PyTorch tensor.
    
    Args:
        audio: Audio array
        device: Target device
        requires_grad: Whether tensor should track gradients
        
    Returns:
        Audio tensor
    """
    tensor = torch.from_numpy(audio.astype(np.float32)).to(device)
    if requires_grad:
        tensor = tensor.requires_grad_(True)
    return tensor


def tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy audio array.
    
    Args:
        tensor: Audio tensor
        
    Returns:
        Audio array
    """
    return tensor.detach().cpu().numpy().astype(np.float32)


def parse_data_file(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Parse data file containing audio files and transcriptions.
    
    Args:
        filepath: Path to data file
        
    Returns:
        List of tuples (audio_file, original_text, target_text)
    """
    try:
        data = np.loadtxt(filepath, dtype=str, delimiter=",")
        
        # Handle different data file formats
        if data.ndim == 1:
            # Single entry
            return [(data[0], data[1], data[2])]
        else:
            # Multiple entries
            result = []
            for i in range(data.shape[1]):
                audio_file = data[0, i]
                original_text = data[1, i]
                target_text = data[2, i]
                result.append((audio_file, original_text, target_text))
            return result
            
    except Exception as e:
        raise RuntimeError(f"Error parsing data file {filepath}: {e}")


def test_ctc_model(model_size: str = 'base', test_audio_path: str = None):
    """
    Test CTC model functionality.
    
    Args:
        model_size: Model size to test
        test_audio_path: Path to test audio file
    """
    print(f"Testing CTC model (size: {model_size})")
    
    # Initialize model
    model = CTCASRModel(model_size=model_size, device='cpu')
    print(f"Model info: {model.get_model_info()}")
    
    if test_audio_path and Path(test_audio_path).exists():
        print(f"Testing transcription on {test_audio_path}")
        
        # Load audio
        audio, sr = load_audio_file(test_audio_path)
        print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
        
        # Test transcription
        transcription = model.transcribe(audio)
        print(f"Transcription: '{transcription}'")
        
        # Test gradient computation
        audio_tensor = audio_to_tensor(audio, requires_grad=True)
        target_text = "hello world"  # Example target
        
        try:
            loss = model.compute_attack_loss(audio_tensor, target_text)
            print(f"Attack loss computation successful: {loss.item():.4f}")
            
            # Test gradient flow
            loss.backward()
            grad_norm = audio_tensor.grad.norm().item()
            print(f"Gradient norm: {grad_norm:.6f}")
            
            if grad_norm > 0:
                print("✅ Gradient flow working correctly!")
            else:
                print("⚠️  No gradients detected")
                
        except Exception as e:
            print(f"❌ Error in gradient computation: {e}")
    
    print("CTC model test completed.")


def test_wav2vec_gradients():
    """Test if Wav2Vec2 model properly computes gradients."""
    print("Testing Wav2Vec2 gradients...")
    model = CTCASRModel(model_size='base')
    
    # Create test audio
    audio = torch.randn(16000, requires_grad=True, device=model.device)
    
    # Compute loss
    loss = model.compute_attack_loss(audio, "hello world")
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    
    if audio.grad is not None:
        grad_norm = audio.grad.norm().item()
        print(f"✅ Wav2Vec2 gradients working! Norm: {grad_norm:.6f}")
        return True
    else:
        print("❌ No Wav2Vec2 gradients detected!")
        return False


if __name__ == "__main__":
    # Test the CTC model
    test_ctc_model('base')
