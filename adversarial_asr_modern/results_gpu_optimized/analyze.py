import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

df = pd.read_csv('./results_full_20250815_012857.csv')

# Basic failure metrics
total_examples = len(df)
successful_attacks = df['success'].sum()
success_rate = successful_attacks / total_examples * 100

# Early termination analysis
completed_runs = (df['success_iteration'] == 1000).sum()
early_terminations = total_examples - completed_runs

# Perturbation saturation
max_pert_reached = (df['max_perturbation'] >= 0.149).sum()
saturation_rate = max_pert_reached / total_examples * 100

print(f"Success Rate: {success_rate:.1f}%")
print(f"Optimization Completion: {completed_runs}/1000")
print(f"Perturbation Saturation: {saturation_rate:.1f}%")
# Loss statistics
loss_stats = df['final_loss'].describe()
print("Final Loss Statistics:")
print(f"Mean: {loss_stats['mean']:.3f} ± {loss_stats['std']:.3f}")
print(f"Range: [{loss_stats['min']:.3f}, {loss_stats['max']:.3f}]")

# Text length analysis
df['orig_length'] = df['original_text'].str.len()
df['target_length'] = df['target_text'].str.len()
df['final_length'] = df['final_text'].str.len()

# Perturbation variance
pert_variance = df['mean_perturbation'].var()
print(f"Perturbation Variance: {pert_variance:.8f}")

# 95% Confidence Interval for success rate (Wilson Score)
from scipy import stats
conf_interval = stats.binom.interval(0.95, total_examples, 0)
conf_interval_pct = [x/total_examples*100 for x in conf_interval]
print(f"95% CI for success rate: [{conf_interval_pct[0]:.2f}%, {conf_interval_pct[1]:.2f}%]")

# Length analysis
import matplotlib.pyplot as plt

length_comparison = pd.DataFrame({
    'Original': df['orig_length'],
    'Target': df['target_length'], 
    'Final': df['final_length']
})

# Plot length distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(['Original', 'Target', 'Final']):
    axes[i].hist(length_comparison[col], bins=30, alpha=0.7)
    axes[i].set_title(f'{col} Text Length Distribution')
    axes[i].set_xlabel('Characters')

# Coherence analysis (manual inspection of samples)
sample_transcriptions = df[['original_text', 'target_text', 'final_text']].sample(10)
print("Sample Transcription Patterns:")
for idx, row in sample_transcriptions.iterrows():
    print(f"\nExample {idx}:")
    print(f"Original: {row['original_text'][:80]}...")
    print(f"Target:   {row['target_text'][:80]}...")
    print(f"Final:    {row['final_text'][:80]}...")


    # If you saved intermediate losses (you might need to re-run a subset)
# Otherwise, analyze final loss patterns

# Final loss analysis
high_loss_examples = df[df['final_loss'] > 6.0]
low_loss_examples = df[df['final_loss'] < 3.0]

print(f"High final loss (>6.0): {len(high_loss_examples)} examples")
print(f"Low final loss (<3.0): {len(low_loss_examples)} examples")

# Loss vs perturbation correlation
correlation = df['final_loss'].corr(df['mean_perturbation'])
print(f"Loss-Perturbation Correlation: {correlation:.3f}")

# Create loss distribution plot
plt.figure(figsize=(10, 6))
plt.hist(df['final_loss'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(df['final_loss'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["final_loss"].mean():.2f}')
plt.xlabel('Final Loss')
plt.ylabel('Frequency')
plt.title('Distribution of Final Attack Losses')
plt.legend()
plt.savefig('loss_distribution.png')
plt.close()

# Save the length distribution figure
fig.savefig('text_length_distributions.png')
plt.close(fig)