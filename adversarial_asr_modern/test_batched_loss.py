#!/usr/bin/env python3
"""
Test script to compare sequential vs batched loss computation performance.
"""

import time
import torch
import numpy as np
from src.adversarial_asr_modern.audio_utils import WhisperASRModel

def test_batched_vs_sequential():
    """Compare performance of sequential vs batched loss computation."""
    
    # Test parameters
    batch_sizes = [1, 5, 10, 20, 50, 100]
    audio_length = 16000 * 3  # 3 seconds
    num_iterations = 5
    
    print("=" * 60)
    print("BATCHED VS SEQUENTIAL LOSS COMPUTATION TEST")
    print("=" * 60)
    print(f"Audio length: {audio_length / 16000:.1f} seconds")
    print(f"Iterations per test: {num_iterations}")
    print()
    
    # Initialize model
    print("Initializing Whisper model...")
    model = WhisperASRModel(device='auto')
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*50}")
        
        # Create test data
        audio_batch = [torch.randn(audio_length, requires_grad=True) for _ in range(batch_size)]
        target_texts = [f"this is test transcription number {i}" for i in range(batch_size)]
        
        # Test sequential processing
        print(f"\n1. SEQUENTIAL (original method):")
        sequential_times = []
        for iteration in range(num_iterations):
            start_time = time.time()
            losses_seq = []
            for i in range(batch_size):
                loss = model.compute_loss(audio_batch[i], target_texts[i])
                losses_seq.append(loss)
            seq_time = time.time() - start_time
            sequential_times.append(seq_time)
            
            if iteration == 0:
                print(f"   First iteration: {seq_time:.3f}s")
                print(f"   Losses computed: {len(losses_seq)}")
        
        avg_seq_time = np.mean(sequential_times)
        print(f"   Average time: {avg_seq_time:.3f}s")
        print(f"   Time per example: {avg_seq_time/batch_size:.3f}s")
        
        # Test batched processing (only for batch_size > 1)
        if batch_size > 1:
            print(f"\n2. BATCHED (single forward pass):")
            batched_times = []
            for iteration in range(num_iterations):
                start_time = time.time()
                losses_batch = model.compute_loss_batch(audio_batch, target_texts)
                batch_time = time.time() - start_time
                batched_times.append(batch_time)
                
                if iteration == 0:
                    print(f"   First iteration: {batch_time:.3f}s")
                    print(f"   Losses computed: {len(losses_batch)}")
            
            avg_batch_time = np.mean(batched_times)
            print(f"   Average time: {avg_batch_time:.3f}s")
            print(f"   Time per example: {avg_batch_time/batch_size:.3f}s")
            
            speedup = avg_seq_time / avg_batch_time
            print(f"\n   🎯 SPEEDUP: {speedup:.2f}x faster")
            
            results.append({
                'batch_size': batch_size,
                'sequential_time': avg_seq_time,
                'batched_time': avg_batch_time,
                'speedup': speedup
            })
        else:
            results.append({
                'batch_size': batch_size,
                'sequential_time': avg_seq_time,
                'batched_time': None,
                'speedup': None
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'Sequential':<12} {'Batched':<12} {'Speedup':<10}")
    print("-" * 50)
    for r in results:
        seq_str = f"{r['sequential_time']:.3f}s"
        batch_str = f"{r['batched_time']:.3f}s" if r['batched_time'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['batch_size']:<12} {seq_str:<12} {batch_str:<12} {speedup_str:<10}")
    
    print("\nRECOMMENDATION:")
    if any(r['speedup'] and r['speedup'] > 5 for r in results):
        print("✅ Batched processing provides MASSIVE speedup!")
        print("   Use batched mode for batch sizes > 10")
    elif any(r['speedup'] and r['speedup'] > 2 for r in results):
        print("✅ Batched processing provides significant speedup!")
        print("   Use batched mode for all batch sizes > 1")
    else:
        print("⚠️  Batched processing provides minimal benefit")
        print("   Consider using sequential processing")

if __name__ == "__main__":
    test_batched_vs_sequential()