#!/usr/bin/env python3
"""
Quick test to verify DataParallel works.
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

from src.adversarial_asr_modern.audio_utils import DataParallelWhisperModel

print("\nInitializing DataParallelWhisperModel...")
model = DataParallelWhisperModel(device='auto')

print("\nTesting with batch_size=2...")
audio_batch = [torch.randn(16000, requires_grad=True) for _ in range(2)]
target_texts = ["hello", "world"]

print("Computing losses...")
losses = model.compute_loss_parallel(audio_batch, target_texts)
print(f"Losses: {[l.item() for l in losses]}")
print("✅ DataParallel test complete!")