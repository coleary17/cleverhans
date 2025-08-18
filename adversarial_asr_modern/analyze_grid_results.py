#!/usr/bin/env python3
"""
Analysis and visualization script for grid search results.
Generates plots and insights from learning rate optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List


def load_grid_results(results_dir: str) -> pd.DataFrame:
    """Load the most recent grid search summary."""
    results_path = Path(results_dir)
    
    # Find most recent summary file
    summary_files = list(results_path.glob("grid_search_summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No grid search summary found in {results_dir}")
    
    latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_summary}")
    
    df = pd.read_csv(latest_summary)
    
    # Clean up any inf values for better visualization
    df['mean_final_loss'] = df['mean_final_loss'].replace([np.inf, -np.inf], np.nan)
    
    return df


def create_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Create heatmap visualizations for different metrics."""
    
    # Prepare data for heatmaps
    metrics = ['success_rate', 'mean_final_loss', 'convergence_rate', 'divergence_ratio']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Grid Search Results: Learning Rate Optimization', fontsize=16, y=1.02)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center')
            ax.set_title(metric.replace('_', ' ').title())
            continue
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values=metric,
            index='lr_stage1',
            columns='lr_stage2',
            aggfunc='mean'  # Average across different initial_bound values
        )
        
        # Handle missing or all-NaN data
        if pivot_data.empty or pivot_data.isna().all().all():
            ax.text(0.5, 0.5, f'No valid {metric} data', ha='center', va='center')
            ax.set_title(metric.replace('_', ' ').title())
            continue
        
        # Choose colormap based on metric
        if metric == 'success_rate':
            cmap = 'YlGn'  # Green for success
            fmt = '.1f'
            vmin, vmax = 0, 100
        elif metric == 'mean_final_loss':
            cmap = 'YlOrRd_r'  # Reversed - lower loss is better
            fmt = '.2f'
            vmin, vmax = None, None
        elif metric == 'convergence_rate':
            cmap = 'RdBu_r'  # Negative (decreasing loss) is better
            fmt = '.3f'
            vmin, vmax = None, None
        else:
            cmap = 'viridis'
            fmt = '.2f'
            vmin, vmax = None, None
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap, 
                   ax=ax, cbar_kws={'label': metric},
                   vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('Learning Rate Stage 2')
        ax.set_ylabel('Learning Rate Stage 1')
        ax.set_title(metric.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_search_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmaps to: {output_dir / 'grid_search_heatmaps.png'}")


def create_scatter_plots(df: pd.DataFrame, output_dir: Path):
    """Create scatter plots to visualize relationships between parameters and metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Relationships and Performance', fontsize=16, y=1.02)
    
    # 1. Success rate vs learning rates
    ax = axes[0, 0]
    scatter = ax.scatter(df['lr_stage1'], df['lr_stage2'], 
                        c=df['success_rate'], s=100, cmap='YlGn', 
                        edgecolors='black', linewidth=1)
    ax.set_xlabel('LR Stage 1')
    ax.set_ylabel('LR Stage 2')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Success Rate by Learning Rates')
    plt.colorbar(scatter, ax=ax, label='Success Rate (%)')
    
    # 2. Mean loss vs learning rates
    ax = axes[0, 1]
    valid_loss = df[df['mean_final_loss'].notna()]
    if not valid_loss.empty:
        scatter = ax.scatter(valid_loss['lr_stage1'], valid_loss['lr_stage2'],
                           c=valid_loss['mean_final_loss'], s=100, cmap='YlOrRd_r',
                           edgecolors='black', linewidth=1)
        ax.set_xlabel('LR Stage 1')
        ax.set_ylabel('LR Stage 2')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Mean Final Loss by Learning Rates')
        plt.colorbar(scatter, ax=ax, label='Mean Loss')
    
    # 3. Success rate vs initial bound
    ax = axes[0, 2]
    for bound in df['initial_bound'].unique():
        bound_data = df[df['initial_bound'] == bound]
        ax.scatter(bound_data['lr_stage1'], bound_data['success_rate'],
                  label=f'Bound={bound}', s=50, alpha=0.7)
    ax.set_xlabel('LR Stage 1')
    ax.set_ylabel('Success Rate (%)')
    ax.set_xscale('log')
    ax.set_title('Success Rate vs LR Stage 1 (by Initial Bound)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Convergence analysis
    ax = axes[1, 0]
    if 'convergence_rate' in df.columns:
        valid_conv = df[df['convergence_rate'].notna()]
        if not valid_conv.empty:
            colors = ['green' if rate < 0 else 'red' for rate in valid_conv['convergence_rate']]
            ax.scatter(valid_conv['lr_stage1'], valid_conv['convergence_rate'],
                      c=colors, s=50, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('LR Stage 1')
            ax.set_ylabel('Convergence Rate')
            ax.set_xscale('log')
            ax.set_title('Convergence Rate (negative = improving)')
            ax.grid(True, alpha=0.3)
    
    # 5. Execution time analysis
    ax = axes[1, 1]
    if 'execution_time' in df.columns:
        ax.scatter(df['success_rate'], df['execution_time'], 
                  c=df['lr_stage1'], s=50, cmap='viridis')
        ax.set_xlabel('Success Rate (%)')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Time vs Success Trade-off')
        ax.grid(True, alpha=0.3)
    
    # 6. Top configurations
    ax = axes[1, 2]
    ax.axis('off')
    
    # Get top 5 configurations
    top_configs = df.nlargest(5, 'success_rate')[['lr_stage1', 'lr_stage2', 
                                                   'initial_bound', 'success_rate', 
                                                   'mean_final_loss']]
    
    text = "Top 5 Configurations:\n\n"
    for idx, row in top_configs.iterrows():
        text += f"{idx+1}. LR1={row['lr_stage1']:.3f}, LR2={row['lr_stage2']:.4f}\n"
        text += f"   Bound={row['initial_bound']:.2f}\n"
        text += f"   Success={row['success_rate']:.1f}%, Loss={row['mean_final_loss']:.2f}\n\n"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_search_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plots to: {output_dir / 'grid_search_scatter.png'}")


def analyze_convergence_patterns(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize convergence patterns from grid search."""
    
    if 'convergence_rate' not in df.columns:
        print("No convergence data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Convergence Pattern Analysis', fontsize=16, y=1.02)
    
    # 1. Distribution of convergence rates
    ax = axes[0, 0]
    valid_conv = df[df['convergence_rate'].notna()]['convergence_rate']
    if not valid_conv.empty:
        ax.hist(valid_conv, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', label='No change')
        ax.set_xlabel('Convergence Rate (loss change per checkpoint)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Convergence Rates')
        ax.legend()
        
        # Add statistics
        stats_text = f"Mean: {valid_conv.mean():.4f}\n"
        stats_text += f"Std: {valid_conv.std():.4f}\n"
        stats_text += f"Converging: {(valid_conv < 0).sum()}/{len(valid_conv)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # 2. Divergence ratio analysis
    ax = axes[0, 1]
    if 'divergence_ratio' in df.columns:
        valid_div = df[df['divergence_ratio'].notna()]
        if not valid_div.empty:
            ax.scatter(valid_div['lr_stage1'], valid_div['divergence_ratio'],
                      c=valid_div['success_rate'], s=50, cmap='RdYlGn')
            ax.set_xlabel('Learning Rate Stage 1')
            ax.set_ylabel('Divergence Ratio')
            ax.set_xscale('log')
            ax.set_title('Divergence Ratio by Learning Rate')
            ax.grid(True, alpha=0.3)
    
    # 3. Initial vs final loss
    ax = axes[1, 0]
    if 'initial_loss' in df.columns and 'final_checkpoint_loss' in df.columns:
        valid_losses = df[df['initial_loss'].notna() & df['final_checkpoint_loss'].notna()]
        if not valid_losses.empty:
            ax.scatter(valid_losses['initial_loss'], valid_losses['final_checkpoint_loss'],
                      c=valid_losses['lr_stage1'], s=50, cmap='viridis')
            
            # Add diagonal line
            max_val = max(valid_losses['initial_loss'].max(), 
                         valid_losses['final_checkpoint_loss'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No change')
            
            ax.set_xlabel('Initial Loss')
            ax.set_ylabel('Final Loss')
            ax.set_title('Loss Progression')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 4. Plateauing analysis
    ax = axes[1, 1]
    if 'plateaued' in df.columns:
        plateau_stats = df.groupby('lr_stage1')['plateaued'].mean() * 100
        plateau_stats.plot(kind='bar', ax=ax)
        ax.set_xlabel('Learning Rate Stage 1')
        ax.set_ylabel('Plateau Rate (%)')
        ax.set_title('Percentage of Configurations that Plateaued')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence analysis to: {output_dir / 'convergence_analysis.png'}")


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate a text report with key findings."""
    
    report = []
    report.append("="*60)
    report.append("GRID SEARCH ANALYSIS REPORT")
    report.append("="*60)
    report.append("")
    
    # Overall statistics
    report.append("1. OVERALL STATISTICS")
    report.append(f"   Total configurations tested: {len(df)}")
    report.append(f"   Configurations with success: {(df['success_rate'] > 0).sum()}")
    report.append(f"   Max success rate: {df['success_rate'].max():.1f}%")
    report.append(f"   Mean success rate: {df['success_rate'].mean():.1f}%")
    report.append("")
    
    # Best configuration
    best_config = df.nlargest(1, 'success_rate').iloc[0]
    report.append("2. BEST CONFIGURATION")
    report.append(f"   Learning Rate Stage 1: {best_config['lr_stage1']}")
    report.append(f"   Learning Rate Stage 2: {best_config['lr_stage2']}")
    report.append(f"   Initial Bound: {best_config['initial_bound']}")
    report.append(f"   Success Rate: {best_config['success_rate']:.1f}%")
    if pd.notna(best_config.get('mean_final_loss')):
        report.append(f"   Mean Final Loss: {best_config['mean_final_loss']:.3f}")
    report.append("")
    
    # Learning rate insights
    report.append("3. LEARNING RATE INSIGHTS")
    
    # Best LR Stage 1
    lr1_success = df.groupby('lr_stage1')['success_rate'].mean()
    best_lr1 = lr1_success.idxmax()
    report.append(f"   Best LR Stage 1: {best_lr1} (avg success: {lr1_success[best_lr1]:.1f}%)")
    
    # Best LR Stage 2
    lr2_success = df.groupby('lr_stage2')['success_rate'].mean()
    best_lr2 = lr2_success.idxmax()
    report.append(f"   Best LR Stage 2: {best_lr2} (avg success: {lr2_success[best_lr2]:.1f}%)")
    
    # Best initial bound
    bound_success = df.groupby('initial_bound')['success_rate'].mean()
    best_bound = bound_success.idxmax()
    report.append(f"   Best Initial Bound: {best_bound} (avg success: {bound_success[best_bound]:.1f}%)")
    report.append("")
    
    # Convergence analysis
    if 'convergence_rate' in df.columns:
        report.append("4. CONVERGENCE ANALYSIS")
        valid_conv = df[df['convergence_rate'].notna()]
        if not valid_conv.empty:
            converging = (valid_conv['convergence_rate'] < 0).sum()
            diverging = (valid_conv['convergence_rate'] > 0).sum()
            report.append(f"   Converging configurations: {converging}/{len(valid_conv)}")
            report.append(f"   Diverging configurations: {diverging}/{len(valid_conv)}")
            report.append(f"   Mean convergence rate: {valid_conv['convergence_rate'].mean():.6f}")
            
            # Find most stable configuration
            valid_conv['stability'] = abs(valid_conv['convergence_rate'])
            most_stable = valid_conv.nsmallest(1, 'stability').iloc[0]
            report.append(f"   Most stable: LR1={most_stable['lr_stage1']}, "
                         f"LR2={most_stable['lr_stage2']} "
                         f"(rate={most_stable['convergence_rate']:.6f})")
        report.append("")
    
    # Failure analysis
    report.append("5. FAILURE ANALYSIS")
    failed = df[df['success_rate'] == 0]
    report.append(f"   Complete failures: {len(failed)}/{len(df)}")
    if len(failed) > 0:
        report.append(f"   Common failure LR1: {failed['lr_stage1'].mode().values[0] if len(failed['lr_stage1'].mode()) > 0 else 'N/A'}")
        report.append(f"   Mean loss for failures: {failed['mean_final_loss'].mean():.3f}")
    
    if 'num_infinite_losses' in df.columns:
        total_inf = df['num_infinite_losses'].sum()
        report.append(f"   Total infinite losses: {total_inf}")
    report.append("")
    
    # Recommendations
    report.append("6. RECOMMENDATIONS")
    report.append("   Based on the grid search results:")
    
    # Check if lower learning rates work better
    if best_lr1 < 0.05:
        report.append("   - Lower learning rates for Stage 1 show better performance")
        report.append("   - Consider testing even lower values (e.g., 0.0001-0.001)")
    elif best_lr1 > 0.05:
        report.append("   - Higher learning rates for Stage 1 work better")
        report.append("   - Current range appears appropriate")
    
    # Check convergence behavior
    if 'convergence_rate' in df.columns:
        avg_conv = df['convergence_rate'].mean()
        if avg_conv > 0:
            report.append("   - Most configurations are diverging - learning rates may be too high")
        elif avg_conv < -0.01:
            report.append("   - Good convergence observed - current approach is working")
        else:
            report.append("   - Slow convergence - consider more iterations or different optimizer")
    
    # Success rate recommendations
    if df['success_rate'].max() == 0:
        report.append("   - No successful attacks - fundamental issues with approach")
        report.append("   - Consider: different optimizer, loss function, or attack strategy")
    elif df['success_rate'].max() < 50:
        report.append("   - Low success rates - significant room for improvement")
        report.append("   - Try: adaptive learning rates, different perturbation bounds")
    else:
        report.append("   - Reasonable success rates achieved")
        report.append("   - Fine-tune the best configuration on larger dataset")
    
    report.append("")
    report.append("="*60)
    
    # Save report
    report_text = "\n".join(report)
    report_file = output_dir / "grid_search_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"Saved report to: {report_file}")
    print("\n" + report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument("--results-dir", type=str, default="grid_search_results",
                       help="Directory containing grid search results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save analysis outputs (default: same as results)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading results from: {args.results_dir}")
    print(f"Saving analysis to: {args.output_dir}")
    
    # Load results
    df = load_grid_results(args.results_dir)
    print(f"Loaded {len(df)} configurations")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_heatmaps(df, output_path)
    create_scatter_plots(df, output_path)
    analyze_convergence_patterns(df, output_path)
    
    # Generate report
    print("\nGenerating analysis report...")
    generate_report(df, output_path)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()