#!/usr/bin/env python3
"""
Simple test to verify batched loss computation works.
"""

import time
import torch
from src.adversarial_asr_modern.audio_utils import WhisperASRModel

print("Initializing Whisper model...")
model = WhisperASRModel(device='auto')

# Test with batch size 2
batch_size = 2
audio_length = 16000 * 2  # 2 seconds

print(f"\nTesting with batch size {batch_size}...")

# Create test data
audio_batch = [torch.randn(audio_length, requires_grad=True) for _ in range(batch_size)]
target_texts = ["hello world", "test transcription"]

# Test sequential
print("\n1. Sequential processing:")
start = time.time()
losses_seq = []
for i in range(batch_size):
    loss = model.compute_loss(audio_batch[i], target_texts[i])
    losses_seq.append(loss)
    print(f"   Example {i}: Loss = {loss.item():.4f}")
seq_time = time.time() - start
print(f"   Total time: {seq_time:.3f}s")

# Test batched
print("\n2. Batched processing:")
start = time.time()
losses_batch = model.compute_loss_batch(audio_batch, target_texts)
batch_time = time.time() - start
for i, loss in enumerate(losses_batch):
    print(f"   Example {i}: Loss = {loss.item():.4f}")
print(f"   Total time: {batch_time:.3f}s")

print(f"\nSpeedup: {seq_time/batch_time:.2f}x")