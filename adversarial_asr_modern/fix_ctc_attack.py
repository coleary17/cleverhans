#!/usr/bin/env python3
"""
Fixed CTC attack that prevents collapse to short outputs.
Uses a combination of CTC loss and length penalty.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.ctc_audio_utils import CTCASRModel, load_audio_file, save_audio_file


def compute_robust_ctc_loss(model, audio_tensor, target_text):
    """
    Compute CTC loss that prevents collapse to short outputs.
    """
    # Get model outputs
    outputs = model.model(input_values=audio_tensor)
    logits = outputs.logits
    
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get target tokens
    with torch.no_grad():
        target_ids = model.processor.tokenizer(target_text, return_tensors="pt").input_ids
        target_ids = target_ids.to(model.device)
    
    # Lengths
    input_lengths = torch.full((1,), logits.shape[1], device=model.device)
    target_lengths = torch.full((1,), target_ids.shape[1], device=model.device)
    
    # Standard CTC loss
    ctc_loss = F.ctc_loss(
        log_probs.transpose(0, 1),
        target_ids,
        input_lengths,
        target_lengths,
        blank=model.processor.tokenizer.pad_token_id,
        zero_infinity=True
    )
    
    # Add length penalty to prevent collapse
    # Penalize if predicted length is much shorter than target
    predicted_tokens = logits.argmax(dim=-1)
    non_blank_mask = predicted_tokens != model.processor.tokenizer.pad_token_id
    predicted_length = non_blank_mask.sum()
    
    length_penalty = torch.abs(predicted_length.float() - target_lengths.float()) * 0.1
    
    # Add entropy bonus to prevent mode collapse
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    entropy_bonus = -entropy * 0.01  # Small negative to encourage diversity
    
    # Combined loss
    total_loss = ctc_loss + length_penalty + entropy_bonus
    
    return total_loss


def test_improved_attack():
    """Test the improved CTC attack."""
    
    print("=" * 60)
    print("Testing Improved CTC Attack")
    print("=" * 60)
    
    # Load test audio
    test_file = "../adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav"
    audio, sr = load_audio_file(test_file, target_sr=16000)
    
    # Initialize model
    model = CTCASRModel(model_size='base', device='auto')
    
    # Get original transcription
    original = model.transcribe(audio)
    print(f"Original: '{original}'")
    
    # Target
    target_text = "OLD WILL IS A FINE FELLOW BUT POOR AND HELPLESS SINCE MISSUS ROGERS HAD HER ACCIDENT"
    print(f"Target:   '{target_text}'")
    
    # Create attack
    audio_tensor = torch.from_numpy(audio).float().to(model.device)
    audio_tensor.requires_grad = True
    
    # Initialize perturbation
    delta = torch.zeros_like(audio_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.01)
    
    print("\nRunning improved attack...")
    print("-" * 60)
    
    for i in range(20):
        optimizer.zero_grad()
        
        # Apply perturbation
        perturbed = (audio_tensor + torch.clamp(delta, -0.2, 0.2)).clamp(-1.0, 1.0)
        
        # Compute robust loss
        loss = compute_robust_ctc_loss(model, perturbed.unsqueeze(0), target_text)
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        
        # Check result
        if i % 5 == 0:
            with torch.no_grad():
                current = model.transcribe(perturbed.cpu().numpy())
                print(f"Step {i:3}: Loss={loss.item():.4f}, Text='{current[:60]}...'")
    
    print("-" * 60)
    print("Test complete!")
    
    # Save final audio
    final_audio = perturbed.detach().cpu().numpy()
    save_audio_file(final_audio, "test_ctc_improved.wav", 16000)
    print(f"Saved to test_ctc_improved.wav")


if __name__ == "__main__":
    test_improved_attack()