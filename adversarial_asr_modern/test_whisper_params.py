#!/usr/bin/env python3
"""
Test script to find optimal parameters for Whisper adversarial attacks.
Helps tune learning rates and bounds to avoid static/noise issues.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import WhisperASRModel, load_audio_file, save_audio_file


def test_perturbation_levels():
    """Test different perturbation levels to find acceptable ranges."""
    
    print("=" * 60)
    print("Testing Whisper with different perturbation levels")
    print("=" * 60)
    
    # Load a test audio file
    test_file = "../adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
    
    if not Path(test_file).exists():
        print(f"Error: Test file not found at {test_file}")
        print("Please ensure LibriSpeech data is available")
        return
    
    # Load audio
    audio, sr = load_audio_file(test_file, target_sr=16000)
    print(f"Loaded audio: shape={audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")
    
    # Initialize model
    model = WhisperASRModel(device='auto')
    
    # Get original transcription
    original_text = model.transcribe(audio)
    print(f"\nOriginal transcription: '{original_text}'")
    
    # Test different perturbation levels
    perturbation_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    print("\n" + "-" * 60)
    print("Testing perturbation levels (uniform noise):")
    print("-" * 60)
    
    for level in perturbation_levels:
        # Add uniform noise
        noise = np.random.uniform(-level, level, size=audio.shape).astype(np.float32)
        perturbed = np.clip(audio + noise, -1.0, 1.0)
        
        # Measure SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Get transcription
        perturbed_text = model.transcribe(perturbed)
        
        # Check if transcription changed
        changed = perturbed_text != original_text
        
        print(f"Level: {level:6.3f} | SNR: {snr_db:6.1f}dB | Changed: {changed:5} | Text: '{perturbed_text[:50]}...'")
        
        # Save example for listening
        if level == 0.02:
            save_audio_file(perturbed, "test_perturbation_0.02.wav", 16000)
            print(f"  → Saved example to test_perturbation_0.02.wav")
    
    print("\n" + "=" * 60)
    print("Recommendations based on testing:")
    print("=" * 60)
    print("For Whisper models:")
    print("  - initial_bound: 0.01 - 0.03 (start conservative)")
    print("  - lr_stage1: 0.0001 - 0.001 (gradient-based optimization)")
    print("  - lr_stage2: 0.00001 - 0.0001 (fine-tuning)")
    print("\nTo avoid static/noise:")
    print("  - Keep perturbations < 0.05 (5% of signal range)")
    print("  - Monitor SNR (should stay > 20dB for imperceptibility)")
    print("  - Use smaller learning rates with more iterations")
    print("\nExample command:")
    print("  uv run python run_attack.py --num-examples 1 --lr-stage1 0.0005 --initial-bound 0.02")


def test_gradient_magnitudes():
    """Test gradient magnitudes to understand appropriate learning rates."""
    
    print("\n" + "=" * 60)
    print("Testing gradient magnitudes for learning rate selection")
    print("=" * 60)
    
    # Load test audio
    test_file = "../adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
    audio, sr = load_audio_file(test_file, target_sr=16000)
    
    # Initialize model
    model = WhisperASRModel(device='auto')
    
    # Create tensor with gradients
    audio_tensor = torch.from_numpy(audio).float().to(model.device)
    audio_tensor.requires_grad = True
    
    # Compute loss with a target phrase
    target_text = "THIS IS A TEST PHRASE"
    loss = model.compute_loss(audio_tensor, target_text)
    
    # Compute gradients
    loss.backward()
    
    # Analyze gradients
    grad = audio_tensor.grad
    grad_norm = grad.norm().item()
    grad_max = grad.abs().max().item()
    grad_mean = grad.abs().mean().item()
    
    print(f"\nGradient statistics:")
    print(f"  - Norm: {grad_norm:.6f}")
    print(f"  - Max: {grad_max:.6f}")
    print(f"  - Mean: {grad_mean:.6f}")
    
    # Suggest learning rates based on gradients
    suggested_lr = 0.01 / grad_norm  # Conservative estimate
    print(f"\nSuggested learning rate: {suggested_lr:.6f}")
    print(f"(Based on 1% step relative to gradient norm)")


if __name__ == "__main__":
    test_perturbation_levels()
    test_gradient_magnitudes()