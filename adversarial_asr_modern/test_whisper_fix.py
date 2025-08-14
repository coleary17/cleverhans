#!/usr/bin/env python3
"""
Test script to verify the Whisper loss fix is working correctly.
This should show the transcription actually changing now.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import WhisperASRModel, load_audio_file


def test_loss_gradient_flow():
    """Test that the loss properly affects transcriptions."""
    
    print("=" * 60)
    print("Testing Fixed Whisper Loss Computation")
    print("=" * 60)
    
    # Load test audio
    test_file = "../adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
    
    if not Path(test_file).exists():
        print(f"Error: Test file not found at {test_file}")
        return
    
    audio, sr = load_audio_file(test_file, target_sr=16000)
    print(f"Loaded audio: shape={audio.shape}")
    
    # Initialize model
    model = WhisperASRModel(device='auto')
    
    # Get original transcription
    original_text = model.transcribe(audio)
    print(f"\nOriginal: '{original_text}'")
    
    # Target text
    target_text = "THIS IS A TEST PHRASE"
    print(f"Target:   '{target_text}'")
    
    # Create tensor with gradients
    audio_tensor = torch.from_numpy(audio).float().to(model.device)
    audio_tensor.requires_grad = True
    
    # Test gradient flow with small perturbation
    print("\n" + "-" * 60)
    print("Testing gradient-based optimization:")
    print("-" * 60)
    
    # Initialize perturbation
    delta = torch.zeros_like(audio_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.01)
    
    # Run a few optimization steps
    for i in range(10):
        optimizer.zero_grad()
        
        # Apply bounded perturbation
        perturbed = (audio_tensor + torch.clamp(delta, -0.05, 0.05)).clamp(-1.0, 1.0)
        
        # Compute loss
        loss = model.compute_loss(perturbed, target_text)
        
        # Backward pass
        loss.backward()
        
        # Check gradient magnitude
        grad_norm = delta.grad.norm().item() if delta.grad is not None else 0
        
        print(f"Step {i+1}: Loss={loss.item():.4f}, Grad norm={grad_norm:.6f}")
        
        # Optimize
        optimizer.step()
        
        # Check transcription every few steps
        if i % 3 == 2:
            with torch.no_grad():
                perturbed_np = perturbed.cpu().numpy()
                current_text = model.transcribe(perturbed_np)
                print(f"  → Current: '{current_text}'")
                
                if current_text != original_text:
                    print(f"  ✓ Transcription changed!")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nIf the transcription is changing, the fix is working!")
    print("If not, there may still be issues with the loss computation.")


if __name__ == "__main__":
    test_loss_gradient_flow()