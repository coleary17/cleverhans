#!/usr/bin/env python3
"""
Grid search script to find optimal learning rates for adversarial attacks.
Tests different learning rate combinations and tracks convergence behavior.
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.adversarial_attack import AdversarialAttack
from adversarial_asr_modern.audio_utils import parse_data_file


class GridSearchConfig:
    """Configuration for grid search parameters."""
    
    # Learning rate ranges
    LR_STAGE1_OPTIONS = [0.001, 0.005, 0.01, 0.05, 0.1]
    LR_STAGE2_OPTIONS = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    INITIAL_BOUND_OPTIONS = [0.05, 0.1, 0.15]
    
    # Fixed parameters for testing
    NUM_EXAMPLES = 20  # Small set for quick testing
    NUM_ITERATIONS_STAGE1 = 500  # Enough to see convergence
    NUM_ITERATIONS_STAGE2 = 100
    BATCH_SIZE = 10  # Smaller batch for detailed tracking
    LOG_INTERVAL = 50  # More frequent logging
    
    # Early stopping parameters
    DIVERGENCE_THRESHOLD = 100  # Stop if loss increases for this many iterations
    MAX_LOSS_THRESHOLD = 50.0  # Stop if loss exceeds this value


def run_single_configuration(config: Dict, data_file: str, config_id: str, 
                             output_dir: Path) -> Dict:
    """
    Run attack with a single configuration and track results.
    
    Args:
        config: Dictionary with lr_stage1, lr_stage2, initial_bound
        data_file: Path to data file
        config_id: Unique identifier for this configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing configuration {config_id}:")
    print(f"  LR Stage 1: {config['lr_stage1']}")
    print(f"  LR Stage 2: {config['lr_stage2']}")
    print(f"  Initial Bound: {config['initial_bound']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Create attack instance
        attack = AdversarialAttack(
            model_name="openai/whisper-base",
            device="auto",
            batch_size=GridSearchConfig.BATCH_SIZE,
            initial_bound=config['initial_bound'],
            lr_stage1=config['lr_stage1'],
            lr_stage2=config['lr_stage2'],
            num_iter_stage1=GridSearchConfig.NUM_ITERATIONS_STAGE1,
            num_iter_stage2=GridSearchConfig.NUM_ITERATIONS_STAGE2,
            log_interval=GridSearchConfig.LOG_INTERVAL,
            verbose=False,
            save_audio=False  # Don't save audio during grid search
        )
        
        # Run attack
        results_file = output_dir / f"results_{config_id}.csv"
        attack.run_attack(
            data_file=data_file,
            root_dir=".",
            output_dir=str(output_dir / "audio"),
            results_file=str(results_file)
        )
        
        # Load and analyze results
        df = pd.read_csv(results_file)
        
        # Load iteration history if available
        history_file = results_file.with_suffix('').with_name(results_file.stem + '_history.json')
        convergence_metrics = {}
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                convergence_metrics = analyze_convergence(history)
        
        # Calculate metrics
        success_rate = df['success'].mean() * 100 if 'success' in df.columns else 0
        
        # Filter out infinite losses for statistics
        finite_losses = df[df['final_loss'] != float('inf')]['final_loss']
        
        metrics = {
            'config_id': config_id,
            'lr_stage1': config['lr_stage1'],
            'lr_stage2': config['lr_stage2'],
            'initial_bound': config['initial_bound'],
            'success_rate': success_rate,
            'num_successes': int(df['success'].sum()) if 'success' in df.columns else 0,
            'mean_final_loss': finite_losses.mean() if len(finite_losses) > 0 else float('inf'),
            'std_final_loss': finite_losses.std() if len(finite_losses) > 0 else 0,
            'min_final_loss': finite_losses.min() if len(finite_losses) > 0 else float('inf'),
            'num_infinite_losses': (df['final_loss'] == float('inf')).sum(),
            'mean_perturbation': df['mean_perturbation'].mean() if 'mean_perturbation' in df.columns else 0,
            'max_perturbation': df['max_perturbation'].max() if 'max_perturbation' in df.columns else 0,
            'execution_time': time.time() - start_time,
            **convergence_metrics
        }
        
        # Check for successful attacks
        if success_rate > 0:
            successful_examples = df[df['success'] == True]
            metrics['avg_success_iteration'] = successful_examples['success_iteration'].mean()
            metrics['first_success_iteration'] = successful_examples['success_iteration'].min()
        else:
            metrics['avg_success_iteration'] = -1
            metrics['first_success_iteration'] = -1
        
        print(f"\nResults for {config_id}:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Mean final loss: {metrics['mean_final_loss']:.3f}")
        print(f"  Execution time: {metrics['execution_time']:.1f}s")
        
        return metrics
        
    except Exception as e:
        print(f"ERROR in configuration {config_id}: {e}")
        return {
            'config_id': config_id,
            'lr_stage1': config['lr_stage1'],
            'lr_stage2': config['lr_stage2'],
            'initial_bound': config['initial_bound'],
            'error': str(e),
            'success_rate': 0,
            'execution_time': time.time() - start_time
        }


def analyze_convergence(history: Dict) -> Dict:
    """
    Analyze convergence behavior from iteration history.
    
    Args:
        history: Iteration history dictionary
        
    Returns:
        Dictionary with convergence metrics
    """
    metrics = {}
    
    if 'losses' not in history or not history['losses']:
        return metrics
    
    # Extract loss trajectories (first 10 examples)
    losses_over_time = []
    for checkpoint_losses in history['losses']:
        # Average loss across examples at this checkpoint
        finite_losses = [l for l in checkpoint_losses if l != float('inf')]
        if finite_losses:
            losses_over_time.append(np.mean(finite_losses))
    
    if len(losses_over_time) < 2:
        return metrics
    
    # Calculate convergence rate (change per checkpoint)
    loss_changes = np.diff(losses_over_time)
    metrics['convergence_rate'] = np.mean(loss_changes)
    metrics['convergence_std'] = np.std(loss_changes)
    
    # Check for divergence (consistent increase)
    diverging_steps = sum(1 for change in loss_changes if change > 0)
    metrics['divergence_ratio'] = diverging_steps / len(loss_changes)
    
    # Loss reduction from start to end
    metrics['total_loss_change'] = losses_over_time[-1] - losses_over_time[0]
    metrics['initial_loss'] = losses_over_time[0]
    metrics['final_checkpoint_loss'] = losses_over_time[-1]
    
    # Check if it plateaued (small changes in last few checkpoints)
    if len(losses_over_time) >= 5:
        recent_changes = loss_changes[-3:]
        metrics['plateaued'] = all(abs(change) < 0.01 for change in recent_changes)
    else:
        metrics['plateaued'] = False
    
    return metrics


def run_grid_search(data_file: str, output_dir: str, parallel: bool = False, 
                    max_workers: int = 4):
    """
    Run grid search over all parameter combinations.
    
    Args:
        data_file: Path to data file
        output_dir: Directory to save results
        parallel: Whether to run configurations in parallel
        max_workers: Number of parallel workers
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate all parameter combinations
    configurations = []
    for lr1, lr2, bound in product(
        GridSearchConfig.LR_STAGE1_OPTIONS,
        GridSearchConfig.LR_STAGE2_OPTIONS,
        GridSearchConfig.INITIAL_BOUND_OPTIONS
    ):
        config = {
            'lr_stage1': lr1,
            'lr_stage2': lr2,
            'initial_bound': bound
        }
        config_id = f"lr1_{lr1}_lr2_{lr2}_bound_{bound}".replace('.', '_')
        configurations.append((config, config_id))
    
    print(f"Total configurations to test: {len(configurations)}")
    print(f"Estimated time: {len(configurations) * 2:.0f}-{len(configurations) * 5:.0f} minutes")
    
    # Run configurations
    results = []
    
    if parallel and max_workers > 1:
        print(f"\nRunning in parallel with {max_workers} workers...")
        with mp.Pool(max_workers) as pool:
            results = pool.starmap(run_single_configuration, 
                                  [(cfg, data_file, cfg_id, output_path) 
                                   for cfg, cfg_id in configurations])
    else:
        print("\nRunning configurations sequentially...")
        for config, config_id in configurations:
            result = run_single_configuration(config, data_file, config_id, output_path)
            results.append(result)
    
    # Save all results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_path / f"grid_search_summary_{timestamp}.csv"
    results_df.to_csv(summary_file, index=False)
    
    # Find best configuration
    if 'success_rate' in results_df.columns:
        # Sort by success rate, then by mean loss
        results_df['score'] = results_df['success_rate'] - results_df['mean_final_loss'].fillna(100) / 100
        best_config = results_df.nlargest(1, 'score').iloc[0]
        
        print("\n" + "="*60)
        print("BEST CONFIGURATION:")
        print("="*60)
        print(f"LR Stage 1: {best_config['lr_stage1']}")
        print(f"LR Stage 2: {best_config['lr_stage2']}")
        print(f"Initial Bound: {best_config['initial_bound']}")
        print(f"Success Rate: {best_config['success_rate']:.1f}%")
        print(f"Mean Loss: {best_config['mean_final_loss']:.3f}")
        
        # Save best configuration
        best_config_dict = {
            'lr_stage1': float(best_config['lr_stage1']),
            'lr_stage2': float(best_config['lr_stage2']),
            'initial_bound': float(best_config['initial_bound']),
            'success_rate': float(best_config['success_rate']),
            'mean_final_loss': float(best_config['mean_final_loss'])
        }
        
        with open(output_path / 'optimal_config.json', 'w') as f:
            json.dump(best_config_dict, f, indent=2)
        
        print(f"\nOptimal configuration saved to: {output_path / 'optimal_config.json'}")
    
    print(f"\nFull results saved to: {summary_file}")
    
    return results_df


def prepare_test_data(num_examples: int = 20) -> str:
    """
    Prepare a small test dataset for grid search.
    
    Args:
        num_examples: Number of examples to use
        
    Returns:
        Path to test data file
    """
    # Check if full data file exists
    if not Path("full_data_flac.txt").exists():
        print("Creating full data file...")
        import subprocess
        subprocess.run(["python", "create_flac_data.py", "--num", "1000"])
    
    # Create test subset
    test_file = f"grid_search_test_{num_examples}.txt"
    
    with open("full_data_flac.txt", 'r') as f:
        lines = f.readlines()
    
    # Select diverse examples (evenly spaced)
    step = len(lines) // num_examples
    selected_lines = [lines[i * step] for i in range(num_examples)]
    
    with open(test_file, 'w') as f:
        f.writelines(selected_lines)
    
    print(f"Created test file with {num_examples} examples: {test_file}")
    return test_file


def main():
    parser = argparse.ArgumentParser(description="Grid search for optimal learning rates")
    parser.add_argument("--data-file", type=str, help="Path to data file")
    parser.add_argument("--num-examples", type=int, default=20, 
                       help="Number of examples for testing")
    parser.add_argument("--output-dir", type=str, default="grid_search_results",
                       help="Directory to save results")
    parser.add_argument("--parallel", action="store_true",
                       help="Run configurations in parallel")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with fewer configurations")
    
    args = parser.parse_args()
    
    # Override config for quick test
    if args.quick:
        GridSearchConfig.LR_STAGE1_OPTIONS = [0.0001, 0.0005]
        GridSearchConfig.LR_STAGE2_OPTIONS = [0.001]
        GridSearchConfig.INITIAL_BOUND_OPTIONS = [0.005, 0.01]
        GridSearchConfig.NUM_ITERATIONS_STAGE1 = 200
        GridSearchConfig.NUM_ITERATIONS_STAGE2 = 50
        GridSearchConfig.NUM_EXAMPLES = 10
        args.num_examples = 10
    
    # Prepare test data
    if not args.data_file:
        GridSearchConfig.NUM_EXAMPLES = args.num_examples
        args.data_file = prepare_test_data(args.num_examples)
    
    print("Starting grid search...")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Run grid search
    results = run_grid_search(
        data_file=args.data_file,
        output_dir=args.output_dir,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    
    print("\nGrid search complete!")


if __name__ == "__main__":
    main()
