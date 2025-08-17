#!/usr/bin/env python3
"""Debug script to test a single adversarial example."""

import sys
sys.path.append('src')

import torch
import numpy as np
from adversarial_asr_modern.audio_utils import WhisperASRModel, load_audio_file
import torch.optim as optim

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

model = WhisperASRModel("openai/whisper-base", device=device)

# Load a single audio file
audio_path = "../adversarial_asr/LibriSpeech/test-clean/61/70968/61-70968-0000.wav"
audio, sr = load_audio_file(audio_path, target_sr=16000)

# Normalize audio
audio = audio.astype(np.float32)
max_val = np.max(np.abs(audio))
if max_val > 1.0:
    audio = audio / max_val

# Convert to tensor
audio_tensor = torch.from_numpy(audio).float().to(device)

# Get original transcription
original_text = model.transcribe(audio)
print(f"Original transcription: {original_text}")

# Set target text
target_text = "THE HAWK SAT ON THE BRANCH"

# Initialize perturbation
delta = torch.zeros_like(audio_tensor, requires_grad=True)
optimizer = optim.Adam([delta], lr=0.05)

print(f"\nTarget text: {target_text}")
print("Starting optimization...")

# Run optimization
for iteration in range(100):
    optimizer.zero_grad()
    
    # Apply perturbation with bounds
    bounded_delta = torch.clamp(delta, -0.15, 0.15)
    perturbed_audio = (audio_tensor + bounded_delta).clamp(-1.0, 1.0)
    
    # Compute loss
    loss = model.compute_loss(perturbed_audio, target_text)
    loss.backward()
    
    # Compare gradient methods
    if iteration == 0:
        print(f"\nGradient stats:")
        print(f"  Max gradient: {delta.grad.max().item():.6f}")
        print(f"  Min gradient: {delta.grad.min().item():.6f}")
        print(f"  Mean abs gradient: {delta.grad.abs().mean().item():.6f}")
    
    # Test both methods
    if iteration < 50:
        # Method 1: Clip gradients (current implementation)
        torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
    else:
        # Method 2: Sign gradients (original paper)
        delta.grad.sign_()
    
    optimizer.step()
    
    if iteration % 20 == 0:
        # Check current transcription
        current_audio = perturbed_audio.detach().cpu().numpy()
        current_text = model.transcribe(current_audio)
        print(f"\n[Iter {iteration}] Loss: {loss.item():.4f}")
        print(f"  Current: {current_text}")
        
        # Check perturbation stats
        pert = bounded_delta.detach()
        print(f"  Max pert: {pert.abs().max().item():.4f}, Mean: {pert.abs().mean().item():.6f}")

print("\n" + "="*50)
print("Analysis complete. The issue is likely in the gradient update method.")
print("The original paper uses signed gradients (FGSM-style) but the current")
print("implementation uses gradient clipping, which may not be effective.")