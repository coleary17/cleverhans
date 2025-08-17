#!/usr/bin/env python3
"""
Test DataParallel performance for Whisper models.
"""

import time
import torch
import numpy as np
from src.adversarial_asr_modern.audio_utils import WhisperASRModel, DataParallelWhisperModel

def test_dataparallel():
    """Compare standard vs DataParallel Whisper performance."""
    
    # Test parameters
    batch_sizes = [1, 5, 10, 20]
    audio_length = 16000 * 3  # 3 seconds
    num_iterations = 5
    
    print("=" * 60)
    print("DATAPARALLEL PERFORMANCE TEST")
    print("=" * 60)
    
    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'auto'
        print("⚠️  CUDA not available, using default device")
    
    print(f"Audio length: {audio_length / 16000:.1f} seconds")
    print(f"Iterations per test: {num_iterations}")
    print()
    
    # Initialize models
    print("Initializing models...")
    print("\n1. Standard WhisperASRModel:")
    standard_model = WhisperASRModel(device=device)
    
    print("\n2. DataParallelWhisperModel:")
    dp_model = DataParallelWhisperModel(device=device)
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*50}")
        
        # Create test data
        audio_batch = [torch.randn(audio_length, requires_grad=True) for _ in range(batch_size)]
        target_texts = [f"test transcription number {i}" for i in range(batch_size)]
        
        # Test standard model
        print(f"\n1. Standard Model (sequential):")
        standard_times = []
        for iteration in range(num_iterations):
            start_time = time.time()
            losses_std = []
            for i in range(batch_size):
                loss = standard_model.compute_loss(audio_batch[i], target_texts[i])
                losses_std.append(loss)
            std_time = time.time() - start_time
            standard_times.append(std_time)
            
            if iteration == 0:
                print(f"   First iteration: {std_time:.3f}s")
                print(f"   Per example: {std_time/batch_size:.3f}s")
        
        avg_std_time = np.mean(standard_times[1:])  # Skip first (warmup)
        print(f"   Average time: {avg_std_time:.3f}s")
        
        # Test DataParallel model
        if batch_size > 1:
            print(f"\n2. DataParallel Model:")
            dp_times = []
            for iteration in range(num_iterations):
                start_time = time.time()
                losses_dp = dp_model.compute_loss_parallel(audio_batch, target_texts)
                dp_time = time.time() - start_time
                dp_times.append(dp_time)
                
                if iteration == 0:
                    print(f"   First iteration: {dp_time:.3f}s")
                    print(f"   Per example: {dp_time/batch_size:.3f}s")
            
            avg_dp_time = np.mean(dp_times[1:])  # Skip first (warmup)
            print(f"   Average time: {avg_dp_time:.3f}s")
            
            speedup = avg_std_time / avg_dp_time
            print(f"\n   🎯 SPEEDUP: {speedup:.2f}x")
            
            results.append({
                'batch_size': batch_size,
                'standard_time': avg_std_time,
                'dataparallel_time': avg_dp_time,
                'speedup': speedup
            })
        else:
            results.append({
                'batch_size': batch_size,
                'standard_time': avg_std_time,
                'dataparallel_time': None,
                'speedup': None
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Batch':<8} {'Standard':<12} {'DataParallel':<14} {'Speedup':<10}")
    print("-" * 50)
    for r in results:
        std_str = f"{r['standard_time']:.3f}s"
        dp_str = f"{r['dataparallel_time']:.3f}s" if r['dataparallel_time'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['batch_size']:<8} {std_str:<12} {dp_str:<14} {speedup_str:<10}")
    
    print("\nRECOMMENDATION:")
    if any(r['speedup'] and r['speedup'] > 1.5 for r in results):
        print("✅ DataParallel provides significant speedup!")
        best = max((r for r in results if r['speedup']), key=lambda x: x['speedup'])
        print(f"   Best speedup: {best['speedup']:.2f}x at batch_size={best['batch_size']}")
    else:
        print("⚠️  DataParallel provides minimal benefit on this hardware")
        print("   Consider using standard sequential processing")

if __name__ == "__main__":
    test_dataparallel()