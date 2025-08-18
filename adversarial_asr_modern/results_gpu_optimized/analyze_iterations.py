import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load both CSV and JSON data
df = pd.read_csv('results_full_20250818_033902.csv')
with open('results_full_20250818_033902_history.json', 'r') as f:
    history = json.load(f)

print("="*60)
print("ITERATION HISTORY ANALYSIS")
print("="*60)

# Analyze available iteration data
num_batches = len(history['losses'])
batch_size = len(history['losses'][0]) if history['losses'] else 0
total_with_history = num_batches * batch_size

print(f"\n1. DATA COVERAGE")
print(f"   Total examples in CSV: {len(df)}")
print(f"   Examples with iteration history: {total_with_history}")
print(f"   Coverage: {total_with_history/len(df)*100:.1f}%")

# Extract loss trajectories for examples with history
print(f"\n2. LOSS CONVERGENCE ANALYSIS")

# Reshape losses into per-example trajectories
loss_trajectories = []
example_idx = 0
for batch_losses in history['losses']:
    for i, loss in enumerate(batch_losses):
        if example_idx not in loss_trajectories:
            loss_trajectories.append([])
        loss_trajectories[example_idx].append(loss)
        example_idx = (example_idx + 1) % batch_size

# Analyze convergence patterns
finite_trajectories = []
infinite_trajectories = []
final_losses = []

for traj in loss_trajectories:
    if any(l == float('inf') for l in traj):
        # Find where it became infinite
        inf_idx = next(i for i, l in enumerate(traj) if l == float('inf'))
        infinite_trajectories.append((traj[:inf_idx], inf_idx))
    else:
        finite_trajectories.append(traj)

print(f"   Trajectories with finite losses: {len(finite_trajectories)}")
print(f"   Trajectories that hit infinity: {len(infinite_trajectories)}")

if infinite_trajectories:
    inf_iterations = [idx * 100 for _, idx in infinite_trajectories]  # Convert to iteration number
    print(f"   Infinity detection iterations: min={min(inf_iterations)}, max={max(inf_iterations)}, mean={np.mean(inf_iterations):.0f}")

# Calculate convergence metrics for finite trajectories
if finite_trajectories:
    print(f"\n3. FINITE TRAJECTORY STATISTICS")
    
    # Filter out empty trajectories
    finite_trajectories = [t for t in finite_trajectories if len(t) > 0]
    
    if not finite_trajectories:
        print("   No valid finite trajectories found")
        final_losses = []
    else:
        # Final losses
        final_losses = [traj[-1] for traj in finite_trajectories]
        print(f"   Final losses: mean={np.mean(final_losses):.3f}, std={np.std(final_losses):.3f}")
        print(f"   Range: [{min(final_losses):.3f}, {max(final_losses):.3f}]")
    
    # Convergence rate (loss reduction per 100 iterations)
    convergence_rates = []
    for traj in finite_trajectories:
        if len(traj) > 1:
            rate = (traj[0] - traj[-1]) / len(traj)
            convergence_rates.append(rate)
    
    if convergence_rates:
        print(f"   Convergence rate: mean={np.mean(convergence_rates):.6f} per iteration")
        print(f"   Total loss reduction: mean={np.mean([traj[0] - traj[-1] for traj in finite_trajectories]):.3f}")
    
    # Check for plateaus (small change in last iterations)
    plateau_threshold = 0.01
    plateaued = 0
    for traj in finite_trajectories:
        if len(traj) >= 5:
            last_5_change = abs(traj[-1] - traj[-5])
            if last_5_change < plateau_threshold:
                plateaued += 1
    
    print(f"   Plateaued trajectories (last 500 iters): {plateaued}/{len(finite_trajectories)} ({plateaued/len(finite_trajectories)*100:.1f}%)")

# Visualize loss trajectories
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. All finite trajectories
ax = axes[0, 0]
iterations = history['iterations']
for i, traj in enumerate(finite_trajectories[:50]):  # Plot first 50 to avoid overcrowding
    ax.plot(iterations[:len(traj)], traj, alpha=0.3, linewidth=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Loss Trajectories (first {min(50, len(finite_trajectories))} finite examples)')
ax.grid(True, alpha=0.3)

# 2. Average trajectory with confidence bands
ax = axes[0, 1]
if finite_trajectories:
    # Pad trajectories to same length
    max_len = max(len(t) for t in finite_trajectories)
    padded = []
    for traj in finite_trajectories:
        padded_traj = traj + [traj[-1]] * (max_len - len(traj))
        padded.append(padded_traj)
    
    padded = np.array(padded)
    mean_traj = np.mean(padded, axis=0)
    std_traj = np.std(padded, axis=0)
    
    iters = iterations[:max_len]
    ax.plot(iters, mean_traj, 'b-', linewidth=2, label='Mean')
    ax.fill_between(iters, mean_traj - std_traj, mean_traj + std_traj, 
                     alpha=0.3, label='±1 std')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Average Loss Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 3. Distribution of final losses
ax = axes[1, 0]
if final_losses:
    ax.hist(final_losses, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_losses), color='red', linestyle='--', 
               label=f'Mean: {np.mean(final_losses):.2f}')
    ax.set_xlabel('Final Loss (after 1000 iterations)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Final Losses (Examples with History)')
    ax.legend()

# 4. Infinity detection timing
ax = axes[1, 1]
if infinite_trajectories:
    inf_iterations = [idx * 100 for _, idx in infinite_trajectories]
    ax.hist(inf_iterations, bins=20, alpha=0.7, edgecolor='black', color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Frequency')
    ax.set_title('When Targets Were Detected as Impossible')
    ax.axvline(np.mean(inf_iterations), color='black', linestyle='--',
               label=f'Mean: {np.mean(inf_iterations):.0f}')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No infinite losses detected', ha='center', va='center')

plt.tight_layout()
plt.savefig('iteration_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n4. VISUALIZATION SAVED: iteration_analysis.png")

# Sample some interesting trajectories
print(f"\n5. INTERESTING EXAMPLES")

if finite_trajectories:
    # Best improvement
    improvements = [(traj[0] - traj[-1], i) for i, traj in enumerate(finite_trajectories)]
    best_improvement, best_idx = max(improvements)
    print(f"\n   Best improvement (Example {best_idx}):")
    print(f"     Start loss: {finite_trajectories[best_idx][0]:.3f}")
    print(f"     Final loss: {finite_trajectories[best_idx][-1]:.3f}")
    print(f"     Improvement: {best_improvement:.3f}")
    
    # Worst final loss
    worst_idx = np.argmax(final_losses)
    print(f"\n   Worst final loss (Example {worst_idx}):")
    print(f"     Start loss: {finite_trajectories[worst_idx][0]:.3f}")
    print(f"     Final loss: {finite_trajectories[worst_idx][-1]:.3f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)