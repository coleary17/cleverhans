#!/usr/bin/env python3
"""
Test script to compare sequential vs parallel processing performance.
"""

import time
import torch
import numpy as np
from src.adversarial_asr_modern.audio_utils import WhisperASRModel, ParallelWhisperASRModel

def test_sequential_vs_parallel():
    """Compare performance of sequential vs parallel processing."""
    
    # Test parameters
    batch_size = 10
    audio_length = 16000 * 3  # 3 seconds
    num_iterations = 10
    
    print("=" * 60)
    print("PARALLEL PROCESSING PERFORMANCE TEST")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Audio length: {audio_length / 16000:.1f} seconds")
    print(f"Iterations: {num_iterations}")
    print()
    
    # Create test data
    print("Creating test data...")
    audio_batch = [torch.randn(audio_length, requires_grad=True) for _ in range(batch_size)]
    target_texts = [f"this is test transcription number {i}" for i in range(batch_size)]
    lengths = [audio_length] * batch_size
    
    # Test sequential processing
    print("\n1. SEQUENTIAL PROCESSING (1 model)")
    print("-" * 40)
    single_model = WhisperASRModel(device='auto')
    
    start_time = time.time()
    for iteration in range(num_iterations):
        losses = []
        for i in range(batch_size):
            loss = single_model.compute_loss(audio_batch[i][:lengths[i]], target_texts[i])
            losses.append(loss)
        
        if iteration == 0:
            print(f"First iteration losses computed: {len(losses)} examples")
    
    sequential_time = time.time() - start_time
    print(f"Total time: {sequential_time:.2f} seconds")
    print(f"Time per iteration: {sequential_time/num_iterations:.2f} seconds")
    print(f"Time per example: {sequential_time/(num_iterations*batch_size):.3f} seconds")
    
    # Test parallel processing with 2 models
    print("\n2. PARALLEL PROCESSING (2 models)")
    print("-" * 40)
    parallel_model_2 = ParallelWhisperASRModel(num_models=2, device='auto')
    
    start_time = time.time()
    for iteration in range(num_iterations):
        losses = parallel_model_2.compute_losses_parallel(audio_batch, target_texts, lengths)
        
        if iteration == 0:
            print(f"First iteration losses computed: {len(losses)} examples")
    
    parallel_time_2 = time.time() - start_time
    print(f"Total time: {parallel_time_2:.2f} seconds")
    print(f"Time per iteration: {parallel_time_2/num_iterations:.2f} seconds")
    print(f"Time per example: {parallel_time_2/(num_iterations*batch_size):.3f} seconds")
    print(f"Speedup vs sequential: {sequential_time/parallel_time_2:.2f}x")
    
    # Test parallel processing with 3 models
    print("\n3. PARALLEL PROCESSING (3 models)")
    print("-" * 40)
    parallel_model_3 = ParallelWhisperASRModel(num_models=3, device='auto')
    
    start_time = time.time()
    for iteration in range(num_iterations):
        losses = parallel_model_3.compute_losses_parallel(audio_batch, target_texts, lengths)
        
        if iteration == 0:
            print(f"First iteration losses computed: {len(losses)} examples")
    
    parallel_time_3 = time.time() - start_time
    print(f"Total time: {parallel_time_3:.2f} seconds")
    print(f"Time per iteration: {parallel_time_3/num_iterations:.2f} seconds")
    print(f"Time per example: {parallel_time_3/(num_iterations*batch_size):.3f} seconds")
    print(f"Speedup vs sequential: {sequential_time/parallel_time_3:.2f}x")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential (1 model):  {sequential_time:.2f}s")
    print(f"Parallel (2 models):   {parallel_time_2:.2f}s ({sequential_time/parallel_time_2:.2f}x speedup)")
    print(f"Parallel (3 models):   {parallel_time_3:.2f}s ({sequential_time/parallel_time_3:.2f}x speedup)")
    
    # Recommendation
    print("\nRECOMMENDATION:")
    if parallel_time_2 < sequential_time * 0.8:
        print("✅ Parallel processing provides significant speedup!")
        print(f"   Use 2-3 parallel models for batch sizes > 5")
    else:
        print("⚠️  Parallel processing overhead may not be worth it for this setup")
        print("   Consider using sequential processing or adjusting parameters")

if __name__ == "__main__":
    test_sequential_vs_parallel()