# Grid Search Usage Guide

## Overview
The grid search system tests different learning rate combinations to find optimal parameters for the adversarial attack. Based on your results showing diverging losses, this will systematically identify better configurations.

## Key Findings from Analysis
- **110 examples in history**: Due to memory optimization, only 10 examples are tracked per checkpoint when batch_size > 20
- **Current issue**: Learning rate 0.1 causes divergence (losses increase from ~10 to ~11)
- **Solution**: Grid search will test lower learning rates to find convergent configurations

## Running Grid Search

### Quick Test (Recommended First)
```bash
# Test with 6 configurations, 5 examples, 200 iterations
$HOME/.local/bin/uv run python grid_search_lr.py --quick

# With parallel execution (faster)
$HOME/.local/bin/uv run python grid_search_lr.py --quick --parallel --max-workers 4
```

### Standard Grid Search
```bash
# Test 75 configurations, 20 examples, 500 iterations
$HOME/.local/bin/uv run python grid_search_lr.py --num-examples 20

# Parallel execution (recommended)
$HOME/.local/bin/uv run python grid_search_lr.py --num-examples 20 --parallel --max-workers 6
```

### Custom Configuration
```bash
# Specify custom parameters
$HOME/.local/bin/uv run python grid_search_lr.py \
    --num-examples 30 \
    --output-dir my_grid_results \
    --parallel \
    --max-workers 8
```

## Analyzing Results

After grid search completes:

```bash
# Generate analysis plots and report
$HOME/.local/bin/uv run python analyze_grid_results.py --results-dir grid_search_results

# For custom output directory
$HOME/.local/bin/uv run python analyze_grid_results.py --results-dir my_grid_results
```

This generates:
- `grid_search_heatmaps.png` - Success rate and loss heatmaps
- `grid_search_scatter.png` - Parameter relationships
- `convergence_analysis.png` - Convergence behavior analysis
- `grid_search_report.txt` - Detailed text report with recommendations
- `optimal_config.json` - Best configuration found

## Parameter Ranges Tested

- **Learning Rate Stage 1**: [0.001, 0.005, 0.01, 0.05, 0.1]
- **Learning Rate Stage 2**: [0.0001, 0.0005, 0.001, 0.005, 0.01]
- **Initial Bound**: [0.05, 0.1, 0.15]

Total: 75 configurations (5 × 5 × 3)

## Expected Improvements

Based on your current results:
1. **Lower learning rates** (0.001-0.01) should prevent divergence
2. **Better convergence** - losses should decrease over iterations
3. **Higher success rates** - current 0% should improve significantly
4. **Stable optimization** - avoid infinite losses

## Time Estimates

- Quick test: ~5-10 minutes
- Standard (20 examples): ~30-60 minutes
- Full grid (75 configs): ~2-5 hours (much faster with parallel execution)

## Using Optimal Configuration

Once found, apply the best configuration:

```python
from adversarial_asr_modern.adversarial_attack import AdversarialAttack
import json

# Load optimal config
with open('grid_search_results/optimal_config.json', 'r') as f:
    config = json.load(f)

# Create attack with optimal parameters
attack = AdversarialAttack(
    model_name="openai/whisper-base",
    device="cuda",
    batch_size=200,
    lr_stage1=config['lr_stage1'],
    lr_stage2=config['lr_stage2'],
    initial_bound=config['initial_bound'],
    num_iter_stage1=1000,
    num_iter_stage2=200
)
```

## Monitoring Progress

The grid search prints progress for each configuration:
- Success rate
- Mean final loss
- Convergence behavior
- Execution time

Failed configurations are logged with error messages for debugging.

## Next Steps

1. Run quick test to validate setup
2. Run standard grid search with parallel execution
3. Analyze results to find optimal configuration
4. Test optimal configuration on full dataset
5. If needed, refine search around best parameters