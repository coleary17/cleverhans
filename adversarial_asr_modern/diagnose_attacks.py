#!/usr/bin/env python3
"""
Diagnostic script to understand why attacks aren't working.
Tests gradients, loss computation, and model behavior.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import WhisperASRModel, load_audio_file


def diagnose_whisper():
    """Diagnose Whisper attack issues."""
    
    print("=" * 60)
    print("WHISPER ATTACK DIAGNOSTICS")
    print("=" * 60)
    
    # Load test audio
    test_file = "../adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
    if not Path(test_file).exists():
        print(f"Error: Test file not found at {test_file}")
        return
        
    audio, sr = load_audio_file(test_file, target_sr=16000)
    print(f"Audio shape: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Initialize model
    model = WhisperASRModel(device='auto')
    
    # Test 1: Basic transcription
    print("\n1. Testing basic transcription:")
    original_text = model.transcribe(audio)
    print(f"   Original: '{original_text}'")
    
    # Test 2: Loss computation
    print("\n2. Testing loss computation:")
    target_text = "THIS IS A TEST"
    audio_tensor = torch.from_numpy(audio).float().to(model.device).requires_grad_(True)
    
    loss = model.compute_loss(audio_tensor, target_text)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 3: Gradient flow
    print("\n3. Testing gradient flow:")
    loss.backward()
    
    if audio_tensor.grad is not None:
        grad_norm = audio_tensor.grad.norm().item()
        grad_max = audio_tensor.grad.abs().max().item()
        grad_mean = audio_tensor.grad.abs().mean().item()
        print(f"   Gradient norm: {grad_norm:.6f}")
        print(f"   Gradient max:  {grad_max:.6f}")
        print(f"   Gradient mean: {grad_mean:.6f}")
    else:
        print("   ERROR: No gradients!")
        
    # Test 4: Optimization test
    print("\n4. Testing optimization (10 steps):")
    audio_tensor = torch.from_numpy(audio).float().to(model.device)
    delta = torch.zeros_like(audio_tensor, requires_grad=True)
    
    # Try different optimizers
    optimizers = [
        ("SGD", torch.optim.SGD([delta], lr=1.0)),
        ("Adam", torch.optim.Adam([delta], lr=0.1)),
    ]
    
    for opt_name, optimizer in optimizers:
        print(f"\n   Using {opt_name}:")
        delta.data.zero_()
        
        for i in range(5):
            optimizer.zero_grad()
            perturbed = (audio_tensor + delta).clamp(-1.0, 1.0)
            loss = model.compute_loss(perturbed, target_text)
            loss.backward()
            
            grad_norm = delta.grad.norm().item() if delta.grad is not None else 0
            optimizer.step()
            
            with torch.no_grad():
                pert_text = model.transcribe(perturbed.cpu().numpy())
                print(f"     Step {i}: Loss={loss.item():.4f}, Grad={grad_norm:.6f}")
                if pert_text != original_text:
                    print(f"     >>> CHANGED TO: '{pert_text[:50]}...'")
    
    # Test 5: Large perturbation test
    print("\n5. Testing with large random perturbation:")
    for noise_level in [0.1, 0.3, 0.5, 1.0]:
        noise = np.random.randn(*audio.shape).astype(np.float32) * noise_level
        perturbed = np.clip(audio + noise, -1.0, 1.0)
        pert_text = model.transcribe(perturbed)
        changed = pert_text != original_text
        print(f"   Noise={noise_level:.1f}: Changed={changed}, Text='{pert_text[:30]}...'")
    
    # Test 6: Check if model is in eval mode
    print("\n6. Model configuration check:")
    print(f"   Model training mode: {model.model.training}")
    print(f"   Requires grad: {any(p.requires_grad for p in model.model.parameters())}")
    print(f"   Device: {model.device}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if grad_norm < 1e-6:
        print("⚠️  ISSUE: Gradients are too small!")
        print("   Solution: Increase learning rate significantly (10x-100x)")
    
    if loss.item() < 0:
        print("⚠️  ISSUE: Loss is negative!")
        print("   Solution: Check loss computation")
    
    if not changed:
        print("⚠️  ISSUE: Model is very robust!")
        print("   Solution: Need larger perturbations or different attack strategy")


if __name__ == "__main__":
    diagnose_whisper()