"""
Audio processing utilities for Whisper-based adversarial attacks.
Modernized replacement for TensorFlow/lingvo-based tool.py.
"""

import torch
import torchaudio
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Union, Tuple, List
import soundfile as sf


class WhisperASRModel:
    """
    Wrapper for OpenAI Whisper model using Hugging Face transformers.
    Replaces the lingvo ASR model functionality.
    """
    
    def __init__(self, model_name="openai/whisper-base", device='auto'):
        # Auto-detect best available device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Auto-detected CUDA GPU available, using: {self.device}")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
                print(f"Auto-detected Apple Silicon GPU (MPS), using: {self.device}")
            else:
                self.device = torch.device('cpu')
                print(f"No GPU detected, using: {self.device}")
        elif device == 'mps':
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
                print(f"Using Apple Silicon GPU (MPS): {self.device}")
            else:
                print(f"Warning: MPS requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            print(f"Using specified device: {self.device}")
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.train()  # Enable gradient computation
        # Keep model weights frozen - we only want input gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Transcribed text as string
        """
        # Ensure audio is the right format for Whisper
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
        # Process audio
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate transcription with forced English
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"], 
                language="en",
                task="transcribe"
            )
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return transcription.strip()
    
    def get_logits(self, audio_array: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Get model logits for audio input (needed for adversarial attacks).
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Model logits as torch.Tensor
        """
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get encoder features
        encoder_outputs = self.model.model.encoder(inputs["input_features"])
        
        return encoder_outputs.last_hidden_state
    
    def compute_mel_spectrogram_differentiable(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram in a differentiable way.
        
        Args:
            audio_tensor: Audio tensor with gradients
            
        Returns:
            Mel-spectrogram tensor with gradients preserved
        """
        # Use torchaudio for differentiable mel-spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            f_min=0,
            f_max=8000
        ).to(self.device)
        
        # Ensure audio is on correct device
        if audio_tensor.device != self.device:
            audio_tensor = audio_tensor.to(self.device)
        
        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Compute mel-spectrogram
        mel_spec = mel_transform(audio_tensor)
        
        # Apply log scaling (Whisper uses log-mel features)
        mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        
        # Normalize to match Whisper's expected input range
        mel_spec = (mel_spec + 4.0) / 4.0  # Approximate normalization
        
        # Pad/trim to match Whisper's expected sequence length (3000 frames for 30s audio)
        target_length = 3000
        if mel_spec.shape[-1] < target_length:
            # Pad with zeros
            padding = target_length - mel_spec.shape[-1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        else:
            # Trim
            mel_spec = mel_spec[:, :, :target_length]
        
        return mel_spec
    
    def compute_loss(self, audio_tensor: torch.Tensor, target_text: str, sample_rate: int = 16000) -> torch.Tensor:
        """
        Compute adversarial loss for Whisper model using CTC-like approach.
        
        Args:
            audio_tensor: Audio data as a tensor with gradients
            target_text: Target transcription text
            sample_rate: Sample rate of audio
            
        Returns:
            Loss tensor
        """
        # Process features while maintaining gradients
        mel_spec = self.compute_mel_spectrogram_differentiable(audio_tensor)
        
        # Get target token IDs
        with torch.no_grad():
            # Encode the target text
            target_encoding = self.processor.tokenizer(
                target_text, 
                return_tensors="pt",
                add_special_tokens=True
            )
            target_ids = target_encoding.input_ids.to(self.device)
        
        # Use the model with labels to get loss directly
        # This is the standard way Whisper computes loss
        outputs = self.model(
            input_features=mel_spec,
            labels=target_ids,
            return_dict=True
        )
        
        # The model returns the loss when labels are provided
        # This loss measures how well the model can produce the target text
        # given the current audio input
        loss = outputs.loss
        
        # Important: To make this work for adversarial attacks,
        # we need to ensure gradients flow back to the input audio
        # The key insight is that we're optimizing the INPUT to minimize
        # the loss of producing the TARGET text
        
        return loss


def create_features(audio_batch: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Create features from audio batch for Whisper model.
    Replaces the lingvo-specific create_features function.
    
    Args:
        audio_batch: Batch of audio tensors [batch_size, audio_length]
        sample_rate: Sample rate of audio
        
    Returns:
        Feature tensor compatible with Whisper
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    
    features_list = []
    for audio in audio_batch:
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        # Resample if necessary
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        # Process with Whisper processor
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        features_list.append(inputs["input_features"])
    
    return torch.cat(features_list, dim=0)


def load_audio_file(filepath: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return audio array and sample rate.
    
    Args:
        filepath: Path to audio file
        target_sr: Target sample rate for resampling
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(filepath, sr=target_sr)
    return audio, sr


def save_audio_file(audio_array: np.ndarray, filepath: str, sample_rate: int = 16000):
    """
    Save audio array to file.
    
    Args:
        audio_array: Audio data as numpy array
        filepath: Output file path
        sample_rate: Sample rate of audio
    """
    sf.write(filepath, audio_array, sample_rate)


def audio_to_tensor(audio_array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy audio array to PyTorch tensor.
    
    Args:
        audio_array: Audio data as numpy array
        device: Target device for tensor
        
    Returns:
        Audio tensor
    """
    return torch.from_numpy(audio_array).float().to(device)


def tensor_to_audio(audio_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy audio array.
    
    Args:
        audio_tensor: Audio tensor
        
    Returns:
        Audio data as numpy array
    """
    return audio_tensor.detach().cpu().numpy()


def apply_reverb(audio_batch: torch.Tensor, rir: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Apply room impulse response (reverb) to audio batch.
    Modernized version of create_speech_rir function.
    
    Args:
        audio_batch: Batch of audio tensors [batch_size, audio_length]
        rir: Room impulse response tensor
        device: Computation device
        
    Returns:
        Audio batch with reverb applied
    """
    device = torch.device(device)
    audio_batch = audio_batch.to(device)
    rir = rir.to(device)
    
    reverb_audio = []
    
    for audio in audio_batch:
        # Convolve with RIR using FFT
        audio_len = audio.shape[0]
        rir_len = rir.shape[0]
        conv_len = audio_len + rir_len - 1
        
        # Pad to power of 2 for efficiency
        fft_len = 2 ** (conv_len - 1).bit_length()
        
        # FFT convolution
        audio_fft = torch.fft.fft(audio, fft_len)
        rir_fft = torch.fft.fft(rir, fft_len)
        conv_result = torch.fft.ifft(audio_fft * rir_fft).real[:conv_len]
        
        # Normalize
        conv_result = conv_result / torch.max(torch.abs(conv_result))
        conv_result = conv_result * (2**15 - 1)  # Scale to int16 range
        conv_result = torch.clamp(conv_result, -2**15, 2**15 - 1)
        
        reverb_audio.append(conv_result[:audio_len])  # Trim to original length
    
    return torch.stack(reverb_audio)


def parse_data_file(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Parse the data file format used in the original project.
    Supports both CSV format and original 3-line format.
    
    Args:
        filepath: Path to data file
        
    Returns:
        List of tuples (audio_file, original_transcription, target_transcription)
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if it's the 3-line format (original read_data_full.txt style)
    if len(lines) == 3:
        # Parse 3-line format
        audio_files = lines[0].strip().split(',')
        original_texts = lines[1].strip().split(',')
        target_texts = lines[2].strip().split(',')
        
        # Combine into tuples
        for i in range(len(audio_files)):
            data.append((
                audio_files[i].strip(),
                original_texts[i].strip() if i < len(original_texts) else "",
                target_texts[i].strip() if i < len(target_texts) else ""
            ))
    else:
        # Parse CSV format (one entry per line)
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                audio_file = parts[0].strip()
                original_trans = parts[1].strip()
                target_trans = parts[2].strip()
                data.append((audio_file, original_trans, target_trans))
    
    return data


def test_whisper_gradients():
    """Test if Whisper model properly computes gradients."""
    print("Testing Whisper gradients...")
    model = WhisperASRModel(device='auto')
    
    # Create test audio with gradients
    audio = torch.randn(16000, requires_grad=True, device=model.device)
    
    # Compute loss
    loss = model.compute_loss(audio, "test transcription")
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    
    if audio.grad is not None:
        grad_norm = audio.grad.norm().item()
        print(f"✅ Whisper gradients working! Norm: {grad_norm:.6f}")
        return True
    else:
        print("❌ No Whisper gradients detected!")
        return False
