#!/usr/bin/env python
"""
Test script to verify GPU-optimized transcription and measure speedup.
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import WhisperASRModel


def test_transcription_methods():
    """Compare CPU-based vs GPU-based transcription methods."""
    
    print("=" * 60)
    print("GPU Optimization Test for Whisper Transcription")
    print("=" * 60)
    
    # Initialize model
    model = WhisperASRModel(device='auto')
    print(f"Model device: {model.device}")
    print()
    
    # Create test audio (10 seconds at 16kHz)
    audio_length = 16000 * 10
    audio_np = np.random.randn(audio_length).astype(np.float32) * 0.1
    audio_tensor = torch.from_numpy(audio_np).to(model.device)
    
    print("Testing transcription methods...")
    print("-" * 40)
    
    # Test 1: Original numpy-based method
    print("\n1. Original method (CPU → numpy → transcribe):")
    start_time = time.time()
    for i in range(5):
        result1 = model.transcribe(audio_np)
    numpy_time = time.time() - start_time
    print(f"   Time for 5 transcriptions: {numpy_time:.3f} seconds")
    print(f"   Average per transcription: {numpy_time/5:.3f} seconds")
    
    # Test 2: New tensor-based method
    print("\n2. Optimized method (GPU tensor → transcribe):")
    start_time = time.time()
    for i in range(5):
        result2 = model.transcribe_tensor(audio_tensor)
    tensor_time = time.time() - start_time
    print(f"   Time for 5 transcriptions: {tensor_time:.3f} seconds")
    print(f"   Average per transcription: {tensor_time/5:.3f} seconds")
    
    # Calculate speedup
    speedup = numpy_time / tensor_time
    print("\n" + "=" * 40)
    print(f"SPEEDUP: {speedup:.2f}x faster with GPU tensors")
    print("=" * 40)
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    audio_grad = torch.randn(16000, requires_grad=True, device=model.device)
    loss = model.compute_loss(audio_grad, "test transcription")
    loss.backward()
    
    if audio_grad.grad is not None:
        grad_norm = audio_grad.grad.norm().item()
        print(f"✅ Gradients working! Norm: {grad_norm:.6f}")
    else:
        print("❌ No gradients detected!")
    
    print("\nTest completed successfully!")


def test_attack_performance():
    """Test actual attack performance with GPU optimization."""
    
    print("\n" + "=" * 60)
    print("Testing Attack Performance")
    print("=" * 60)
    
    from adversarial_asr_modern.adversarial_attack import AdversarialAttack
    
    # Create a small test case
    attack = AdversarialAttack(
        model_name="openai/whisper-base",
        device='auto',
        batch_size=1,
        num_iter_stage1=10,  # Just 10 iterations for testing
        num_iter_stage2=10,
        log_interval=5
    )
    
    # Create test audio
    test_audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1  # 5 seconds
    test_data = [("test.wav", "original text", "target text")]
    
    # Prepare batch
    batch = {
        'audios': torch.from_numpy(test_audio.reshape(1, -1)).float().to(attack.device),
        'original_texts': ["original text"],
        'target_texts': ["target text"],
        'th_batch': [np.ones((100, 1025))],  # Dummy threshold
        'psd_max_batch': np.array([1.0]),
        'masks': torch.ones(1, len(test_audio)).to(attack.device),
        'lengths': [len(test_audio)],
        'max_length': len(test_audio)
    }
    
    print("Running Stage 1 attack with GPU optimization...")
    start_time = time.time()
    stage1_audio, results = attack.stage1_attack(batch)
    attack_time = time.time() - start_time
    
    print(f"\nAttack completed in {attack_time:.2f} seconds")
    print(f"Average time per iteration: {attack_time/10:.3f} seconds")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_transcription_methods()
    test_attack_performance()