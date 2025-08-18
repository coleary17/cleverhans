import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_csv('/Users/conor/Dev/thesis/cleverhans/adversarial_asr_modern/results_gpu_optimized/results_full_20250818_033902.csv')

print("="*60)
print("ADVERSARIAL ATTACK RESULTS ANALYSIS")
print("="*60)

# Basic metrics
total_examples = len(df)
successful_attacks = df['success'].sum()
success_rate = successful_attacks / total_examples * 100

print(f"\n1. SUCCESS METRICS")
print(f"   Total Examples: {total_examples}")
print(f"   Successful Attacks: {successful_attacks}")
print(f"   Success Rate: {success_rate:.1f}%")

# Handle infinity values
df['final_loss_clean'] = df['final_loss'].replace([np.inf, -np.inf], np.nan)
finite_losses = df[df['final_loss_clean'].notna()]
infinite_losses = df[df['final_loss'] == np.inf]

print(f"\n2. LOSS ANALYSIS")
print(f"   Examples with finite loss: {len(finite_losses)}")
print(f"   Examples with infinite loss: {len(infinite_losses)}")

if len(finite_losses) > 0:
    loss_stats = finite_losses['final_loss_clean'].describe()
    print(f"   Finite Loss Statistics:")
    print(f"     Mean: {loss_stats['mean']:.3f} ± {loss_stats['std']:.3f}")
    print(f"     Median: {loss_stats['50%']:.3f}")
    print(f"     Range: [{loss_stats['min']:.3f}, {loss_stats['max']:.3f}]")

# Analyze reasons for failure
if 'reason' in df.columns:
    reason_counts = df['reason'].value_counts(dropna=False)
    print(f"\n3. FAILURE REASONS")
    for reason, count in reason_counts.items():
        if pd.isna(reason):
            reason = "No specific reason (normal termination)"
        print(f"   {reason}: {count} ({count/total_examples*100:.1f}%)")

# Early termination analysis
completed_runs = (df['success_iteration'] == 1000).sum()
early_terminations = total_examples - completed_runs

print(f"\n4. OPTIMIZATION ANALYSIS")
print(f"   Completed full 1000 iterations: {completed_runs}")
print(f"   Early terminations: {early_terminations}")

# Perturbation analysis
max_pert_reached = (df['max_perturbation'] >= 0.149).sum()
saturation_rate = max_pert_reached / total_examples * 100

print(f"\n5. PERTURBATION ANALYSIS")
print(f"   Max perturbation reached (≥0.149): {max_pert_reached} ({saturation_rate:.1f}%)")
print(f"   Mean perturbation statistics:")
print(f"     Mean: {df['mean_perturbation'].mean():.6f}")
print(f"     Std: {df['mean_perturbation'].std():.6f}")
print(f"     Range: [{df['mean_perturbation'].min():.6f}, {df['mean_perturbation'].max():.6f}]")

# Text length analysis
df['orig_length'] = df['original_text'].str.len()
df['target_length'] = df['target_text'].str.len()
df['final_length'] = df['final_text'].fillna('').str.len()

print(f"\n6. TEXT LENGTH ANALYSIS")
print(f"   Original text length: {df['orig_length'].mean():.1f} ± {df['orig_length'].std():.1f}")
print(f"   Target text length: {df['target_length'].mean():.1f} ± {df['target_length'].std():.1f}")
print(f"   Final text length: {df['final_length'].mean():.1f} ± {df['final_length'].std():.1f}")

# Correlation analysis for finite losses
if len(finite_losses) > 0:
    correlation = finite_losses['final_loss_clean'].corr(finite_losses['mean_perturbation'])
    print(f"\n7. CORRELATION ANALYSIS")
    print(f"   Loss-Perturbation Correlation: {correlation:.3f}")

# Sample analysis
print(f"\n8. SAMPLE TRANSCRIPTIONS")
print("   Examples with finite loss:")
if len(finite_losses) > 0:
    sample_finite = finite_losses.nsmallest(3, 'final_loss_clean')
    for idx, row in sample_finite.iterrows():
        print(f"\n   Example {idx} (Loss: {row['final_loss_clean']:.3f}):")
        print(f"     Original: {row['original_text'][:60]}...")
        print(f"     Target:   {row['target_text'][:60]}...")
        print(f"     Final:    {row['final_text'][:60] if pd.notna(row['final_text']) else '[No transcription]'}...")

print("\n   Examples with infinite loss (first 3):")
if len(infinite_losses) > 0:
    sample_infinite = infinite_losses.head(3)
    for idx, row in sample_infinite.iterrows():
        print(f"\n   Example {idx}:")
        print(f"     Original: {row['original_text'][:60]}...")
        print(f"     Target:   {row['target_text'][:60]}...")
        print(f"     Reason:   {row['reason'] if pd.notna(row['reason']) else 'Unknown'}")

# Create visualizations with proper handling of inf values
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Loss distribution (finite values only)
if len(finite_losses) > 0:
    axes[0, 0].hist(finite_losses['final_loss_clean'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(finite_losses['final_loss_clean'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {finite_losses["final_loss_clean"].mean():.2f}')
    axes[0, 0].set_xlabel('Final Loss')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of Finite Losses (n={len(finite_losses)})')
    axes[0, 0].legend()
else:
    axes[0, 0].text(0.5, 0.5, 'No finite loss values', ha='center', va='center')
    axes[0, 0].set_title('Distribution of Finite Losses')

# 2. Perturbation distribution
axes[0, 1].hist(df['mean_perturbation'], bins=30, alpha=0.7, edgecolor='black', color='green')
axes[0, 1].axvline(df['mean_perturbation'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["mean_perturbation"].mean():.4f}')
axes[0, 1].set_xlabel('Mean Perturbation')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Mean Perturbations')
axes[0, 1].legend()

# 3. Success iteration distribution
success_iterations = df[df['success_iteration'] > 0]['success_iteration']
if len(success_iterations) > 0:
    axes[1, 0].hist(success_iterations, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Success Iterations')
else:
    axes[1, 0].text(0.5, 0.5, 'No successful attacks', ha='center', va='center')
    axes[1, 0].set_title('Distribution of Success Iterations')

# 4. Text length comparison
length_data = [df['orig_length'], df['target_length'], df['final_length']]
bp = axes[1, 1].boxplot(length_data, labels=['Original', 'Target', 'Final'])
axes[1, 1].set_ylabel('Text Length (characters)')
axes[1, 1].set_title('Text Length Comparison')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attack_analysis_fixed.png', dpi=150, bbox_inches='tight')
print(f"\n9. VISUALIZATION SAVED")
print(f"   Output: attack_analysis_fixed.png")

# Additional analysis for impossible targets
impossible_targets = df[df['reason'] == 'impossible_target']
if len(impossible_targets) > 0:
    print(f"\n10. IMPOSSIBLE TARGET ANALYSIS")
    print(f"   Total impossible targets: {len(impossible_targets)}")
    print(f"   Examples of impossible targets:")
    for idx, row in impossible_targets.head(3).iterrows():
        print(f"     Target: {row['target_text'][:80]}...")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)