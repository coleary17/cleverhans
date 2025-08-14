#!/usr/bin/env python3
"""
Exact implementation of the original attack algorithm.
Uses signed gradients and adaptive rescaling.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import WhisperASRModel, load_audio_file, save_audio_file


def exact_attack_algorithm():
    """Implement the exact attack from the original paper."""
    
    print("=" * 70)
    print("EXACT ATTACK ALGORITHM FROM ORIGINAL PAPER")
    print("=" * 70)
    
    # Load audio
    test_file = "../adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
    audio, sr = load_audio_file(test_file, target_sr=16000)
    
    # Initialize model
    model = WhisperASRModel(device='auto')
    
    # Get original transcription
    original_text = model.transcribe(audio)
    print(f"Original: '{original_text}'")
    
    # Target
    target_text = "OLD WILL IS A FINE FELLOW BUT POOR AND HELPLESS SINCE MISSUS ROGERS HAD HER ACCIDENT"
    print(f"Target:   '{target_text}'")
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float().to(model.device)
    
    # CRITICAL PARAMETERS (scaled from original)
    INT16_MAX = 32768.0
    initial_bound = 2000.0 / INT16_MAX  # 0.061
    lr = 100.0 / INT16_MAX  # 0.003
    
    # But actually, let's try MUCH higher for Whisper
    initial_bound = 0.3  # 30% of signal
    lr = 1.0  # High learning rate
    
    print(f"\nParameters:")
    print(f"  Initial bound: {initial_bound}")
    print(f"  Learning rate: {lr}")
    print("-" * 70)
    
    # Initialize perturbation
    delta = torch.zeros_like(audio_tensor, requires_grad=True)
    
    # Rescale factor (key from original)
    rescale = 1.0
    
    # Use SGD with momentum (simpler than Adam for debugging)
    optimizer = torch.optim.SGD([delta], lr=lr, momentum=0.9)
    
    print("\nRunning attack...")
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Apply bounded perturbation with rescaling (KEY!)
        bounded_delta = torch.clamp(delta, -initial_bound, initial_bound)
        scaled_delta = bounded_delta * rescale
        perturbed = (audio_tensor + scaled_delta).clamp(-1.0, 1.0)
        
        # Compute loss
        loss = model.compute_loss(perturbed, target_text)
        
        # Backward
        loss.backward()
        
        # CRITICAL: Sign the gradients (from original paper)
        if delta.grad is not None:
            grad_norm = delta.grad.norm().item()
            
            # Use signed gradients
            delta.grad = delta.grad.sign()
            
            # Alternative: scale gradients
            # if grad_norm > 0:
            #     delta.grad = delta.grad / grad_norm
        
        # Update
        optimizer.step()
        
        # Check result every 10 iterations
        if i % 10 == 0:
            with torch.no_grad():
                current_text = model.transcribe(perturbed.cpu().numpy())
                max_pert = scaled_delta.abs().max().item()
                
                print(f"Iter {i:3}: Loss={loss.item():.4f}, Max pert={max_pert:.4f}")
                print(f"          Text: '{current_text[:60]}...'")
                
                # Check for success
                if current_text.lower().strip() == target_text.lower().strip():
                    print(f"\n SUCCESS at iteration {i}!")
                    
                    # Adaptive rescaling (from original)
                    current_max = bounded_delta.abs().max().item()
                    if rescale * initial_bound > current_max:
                        rescale = current_max / initial_bound
                    rescale *= 0.8
                    print(f"Rescaling to {rescale:.4f}")
                    
                    # Save successful audio
                    save_audio_file(perturbed.cpu().numpy(), f"success_iter_{i}.wav", 16000)
                    break
                    
                # If text changed at all, that's progress
                elif current_text != original_text:
                    print(f"          >>> TEXT CHANGED!")
    
    print("-" * 70)
    print("Attack complete!")
    
    # Save final audio
    final_audio = perturbed.detach().cpu().numpy()
    save_audio_file(final_audio, "exact_algorithm_output.wav", 16000)
    print(f"Saved to exact_algorithm_output.wav")


if __name__ == "__main__":
    exact_attack_algorithm()