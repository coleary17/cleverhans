import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from .audio_utils import (
    WhisperASRModel, ParallelWhisperASRModel, DataParallelWhisperModel,
    load_audio_file, save_audio_file, 
    audio_to_tensor, tensor_to_audio, parse_data_file
)
from .masking_threshold import generate_th, Transform

class AttackDiagnostics:
    """Complete diagnostic suite for understanding attack failure"""
    
    def __init__(self, model, processor, device='cpu'):
        self.model = model
        self.processor = processor
        self.device = device
        
    def verify_loss_implementation(self, audio_samples, target_texts, n_samples=10):
        """Verify your loss matches Whisper's native loss"""
        print("=" * 50)
        print("LOSS VERIFICATION TEST")
        print("=" * 50)
        
        differences = []
        
        for i, (audio, target) in enumerate(zip(audio_samples[:n_samples], target_texts[:n_samples])):
            # Prepare inputs
            inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
            input_features = inputs.input_features.to(self.device)
            
            labels = self.processor.tokenizer(target, return_tensors="pt").input_ids.to(self.device)
            
            # Whisper's native loss
            outputs = self.model(input_features=input_features, labels=labels)
            native_loss = outputs.loss.item()
            
            # Your manual loss computation
            logits = outputs.logits
            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = labels[0, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction='mean', 
                ignore_index=self.processor.tokenizer.pad_token_id
            )
            manual_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            ).item()
            
            diff = abs(native_loss - manual_loss)
            differences.append(diff)
            
            print(f"Sample {i+1}: Native={native_loss:.4f}, Manual={manual_loss:.4f}, Diff={diff:.6f}")
        
        avg_diff = np.mean(differences)
        print(f"\n✓ Average difference: {avg_diff:.6f}")
        
        if avg_diff < 0.001:
            print("✓ Loss implementations match perfectly!")
        elif avg_diff < 0.01:
            print("✓ Loss implementations are effectively identical")
        else:
            print("⚠ Significant difference detected - investigate further")
            
        return differences
    
    def analyze_gradient_flow(self, audio, target_text, num_iterations=100):
        """Analyze how gradients behave during attack"""
        print("\n" + "=" * 50)
        print("GRADIENT FLOW ANALYSIS")
        print("=" * 50)
        
        # Prepare inputs
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features.to(self.device).requires_grad_(True)
        labels = self.processor.tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)
        
        gradient_norms = []
        losses = []
        gradient_stats = defaultdict(list)
        
        # Run abbreviated attack
        perturbed = input_features.clone().requires_grad_(True)
        
        for i in range(num_iterations):
            outputs = self.model(input_features=perturbed, labels=labels)
            loss = outputs.loss
            
            # Compute gradients
            loss.backward()
            
            # Collect statistics
            grad = perturbed.grad
            gradient_norms.append(grad.norm().item())
            losses.append(loss.item())
            
            # Detailed gradient statistics
            gradient_stats['mean'].append(grad.mean().item())
            gradient_stats['std'].append(grad.std().item())
            gradient_stats['max'].append(grad.abs().max().item())
            gradient_stats['min'].append(grad.abs().min().item())
            
            # Frequency analysis (for spectrograms)
            grad_np = grad.detach().cpu().numpy().squeeze()
            if len(grad_np.shape) == 2:  # [freq, time]
                low_freq = np.mean(np.abs(grad_np[:30, :]))
                mid_freq = np.mean(np.abs(grad_np[30:60, :]))
                high_freq = np.mean(np.abs(grad_np[60:, :]))
                gradient_stats['low_freq'].append(low_freq)
                gradient_stats['mid_freq'].append(mid_freq)
                gradient_stats['high_freq'].append(high_freq)
            
            # Update perturbation
            with torch.no_grad():
                perturbed = perturbed - 0.01 * grad.sign()
                perturbed = perturbed.detach().requires_grad_(True)
        
        # Print summary
        print(f"Initial gradient norm: {gradient_norms[0]:.4f}")
        print(f"Final gradient norm: {gradient_norms[-1]:.4f}")
        print(f"Gradient decay ratio: {gradient_norms[-1]/gradient_norms[0]:.6f}")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        
        return {
            'gradient_norms': gradient_norms,
            'losses': losses,
            'gradient_stats': dict(gradient_stats)
        }
    
    def analyze_attention_patterns(self, audio, target_text):
        """Analyze attention behavior during attack"""
        print("\n" + "=" * 50)
        print("ATTENTION PATTERN ANALYSIS")
        print("=" * 50)
        
        # Get clean and perturbed inputs
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
        clean_input = inputs.input_features.to(self.device)
        
        # Create adversarial perturbation (simplified)
        perturbed_input = clean_input.clone().requires_grad_(True)
        labels = self.processor.tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)
        
        # Run a few attack iterations
        for _ in range(50):
            outputs = self.model(input_features=perturbed_input, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            with torch.no_grad():
                perturbed_input = perturbed_input - 0.01 * perturbed_input.grad.sign()
                perturbed_input = perturbed_input.detach().requires_grad_(True)
        
        # Compare outputs
        with torch.no_grad():
            clean_output = self.model.generate(clean_input)
            adv_output = self.model.generate(perturbed_input)
            
            clean_text = self.processor.decode(clean_output[0], skip_special_tokens=True)
            adv_text = self.processor.decode(adv_output[0], skip_special_tokens=True)
            
        print(f"Original: '{clean_text}'")
        print(f"After attack: '{adv_text}'")
        print(f"Target: '{target_text}'")
        print(f"Attack {'SUCCEEDED' if adv_text == target_text else 'FAILED'}")
        
        # Calculate perturbation statistics
        perturbation = (perturbed_input - clean_input).detach().cpu().numpy()
        
        return {
            'original_text': clean_text,
            'adversarial_text': adv_text,
            'target_text': target_text,
            'success': adv_text == target_text,
            'perturbation_l2': np.linalg.norm(perturbation),
            'perturbation_linf': np.abs(perturbation).max()
        }
    
    def analyze_loss_landscape(self, audio, target_text, num_points=20):
        """Visualize loss landscape around clean input"""
        print("\n" + "=" * 50)
        print("LOSS LANDSCAPE ANALYSIS")
        print("=" * 50)
        
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features.to(self.device)
        labels = self.processor.tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)
        
        # Generate two random directions
        dir1 = torch.randn_like(input_features)
        dir1 = dir1 / dir1.norm()
        
        dir2 = torch.randn_like(input_features)
        dir2 = dir2 / dir2.norm()
        
        # Sample loss landscape
        alphas = np.linspace(-0.1, 0.1, num_points)
        betas = np.linspace(-0.1, 0.1, num_points)
        
        loss_surface = np.zeros((num_points, num_points))
        
        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    perturbed = input_features + alpha * dir1 + beta * dir2
                    outputs = self.model(input_features=perturbed, labels=labels)
                    loss_surface[i, j] = outputs.loss.item()
        
        # Calculate landscape statistics
        flatness = np.std(loss_surface) / np.mean(loss_surface)
        
        print(f"Loss landscape statistics:")
        print(f"  Mean loss: {np.mean(loss_surface):.4f}")
        print(f"  Std dev: {np.std(loss_surface):.4f}")
        print(f"  Min/Max: {np.min(loss_surface):.4f} / {np.max(loss_surface):.4f}")
        print(f"  Flatness metric: {flatness:.4f} (lower = flatter)")
        
        return {
            'loss_surface': loss_surface,
            'flatness': flatness,
            'mean_loss': np.mean(loss_surface),
            'std_loss': np.std(loss_surface)
        }
    
    def run_complete_diagnostics(self, audio_samples, target_texts, n_samples=3):
        """Run all diagnostics and create visualizations"""
        
        print("\n" + "=" * 60)
        print("RUNNING COMPLETE DIAGNOSTIC SUITE")
        print("=" * 60)
        
        # 1. Verify loss implementation
        loss_diffs = self.verify_loss_implementation(audio_samples, target_texts)
        
        all_results = []
        
        # 2. Run detailed analysis on a few samples
        for i in range(min(n_samples, len(audio_samples))):
            print(f"\n{'='*60}")
            print(f"ANALYZING SAMPLE {i+1}/{n_samples}")
            print(f"{'='*60}")
            
            audio = audio_samples[i]
            target = target_texts[i]
            
            # Gradient flow analysis
            gradient_results = self.analyze_gradient_flow(audio, target, num_iterations=100)
            
            # Attention analysis
            attention_results = self.analyze_attention_patterns(audio, target)
            
            # Loss landscape (only for first sample to save time)
            if i == 0:
                landscape_results = self.analyze_loss_landscape(audio, target, num_points=15)
            else:
                landscape_results = None
            
            all_results.append({
                'sample_id': i,
                'gradient_analysis': gradient_results,
                'attention_analysis': attention_results,
                'landscape_analysis': landscape_results
            })
            
        # 3. Create visualizations
        self.create_diagnostic_plots(all_results)
        
        # 4. Save results
        self.save_results(all_results)
        
        return all_results
    
    def create_diagnostic_plots(self, results):
        """Create comprehensive diagnostic visualizations"""
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Plot 1: Gradient norms across samples
        for i, res in enumerate(results):
            if 'gradient_analysis' in res:
                axes[0, 0].plot(res['gradient_analysis']['gradient_norms'], 
                               label=f'Sample {i+1}', alpha=0.7)
        axes[0, 0].set_title('Gradient Norm Evolution')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('L2 Norm')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        
        # Plot 2: Loss evolution
        for i, res in enumerate(results):
            if 'gradient_analysis' in res:
                axes[0, 1].plot(res['gradient_analysis']['losses'], 
                               label=f'Sample {i+1}', alpha=0.7)
        axes[0, 1].set_title('Loss Evolution')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cross-Entropy Loss')
        axes[0, 1].legend()
        
        # Plot 3: Frequency-wise gradients (first sample)
        if results[0]['gradient_analysis']['gradient_stats'].get('low_freq'):
            stats = results[0]['gradient_analysis']['gradient_stats']
            axes[0, 2].plot(stats['low_freq'], label='Low Freq', alpha=0.7)
            axes[0, 2].plot(stats['mid_freq'], label='Mid Freq', alpha=0.7)
            axes[0, 2].plot(stats['high_freq'], label='High Freq', alpha=0.7)
            axes[0, 2].set_title('Gradient by Frequency Band')
            axes[0, 2].set_xlabel('Iteration')
            axes[0, 2].set_ylabel('Mean Gradient Magnitude')
            axes[0, 2].legend()
        
        # Plot 4: Gradient statistics
        if results[0]['gradient_analysis']:
            stats = results[0]['gradient_analysis']['gradient_stats']
            axes[1, 0].plot(stats['std'], label='Std Dev', alpha=0.7)
            axes[1, 0].plot(np.abs(stats['mean']), label='|Mean|', alpha=0.7)
            axes[1, 0].set_title('Gradient Statistics')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
        
        # Plot 5: Loss landscape (if computed)
        if results[0].get('landscape_analysis') and results[0]['landscape_analysis']:
            landscape = results[0]['landscape_analysis']['loss_surface']
            im = axes[1, 1].imshow(landscape, cmap='viridis', aspect='auto')
            axes[1, 1].set_title(f"Loss Landscape (Flatness: {results[0]['landscape_analysis']['flatness']:.3f})")
            axes[1, 1].set_xlabel('Direction 1')
            axes[1, 1].set_ylabel('Direction 2')
            plt.colorbar(im, ax=axes[1, 1])
        
        # Plot 6: Success summary
        success_data = [1 if r['attention_analysis']['success'] else 0 for r in results]
        axes[1, 2].bar(range(len(success_data)), success_data)
        axes[1, 2].set_title('Attack Success by Sample')
        axes[1, 2].set_xlabel('Sample ID')
        axes[1, 2].set_ylabel('Success (1) / Failure (0)')
        axes[1, 2].set_ylim([0, 1.2])
        
        # Plot 7: Perturbation magnitudes
        l2_norms = [r['attention_analysis']['perturbation_l2'] for r in results]
        linf_norms = [r['attention_analysis']['perturbation_linf'] for r in results]
        x = np.arange(len(l2_norms))
        width = 0.35
        axes[2, 0].bar(x - width/2, l2_norms, width, label='L2 Norm')
        axes[2, 0].bar(x + width/2, linf_norms, width, label='L∞ Norm')
        axes[2, 0].set_title('Perturbation Magnitudes')
        axes[2, 0].set_xlabel('Sample ID')
        axes[2, 0].set_ylabel('Norm')
        axes[2, 0].legend()
        
        # Plot 8: Gradient decay comparison
        decay_ratios = []
        for res in results:
            if 'gradient_analysis' in res:
                norms = res['gradient_analysis']['gradient_norms']
                if len(norms) > 0 and norms[0] > 0:
                    decay_ratios.append(norms[-1] / norms[0])
        
        if decay_ratios:
            axes[2, 1].bar(range(len(decay_ratios)), decay_ratios)
            axes[2, 1].set_title('Gradient Decay Ratio (Final/Initial)')
            axes[2, 1].set_xlabel('Sample ID')
            axes[2, 1].set_ylabel('Decay Ratio')
            axes[2, 1].set_yscale('log')
        
        # Plot 9: Text length comparison
        for i, res in enumerate(results):
            orig_len = len(res['attention_analysis']['original_text'].split())
            adv_len = len(res['attention_analysis']['adversarial_text'].split())
            target_len = len(res['attention_analysis']['target_text'].split())
            
            x = [i*3, i*3+0.5, i*3+1]
            y = [orig_len, adv_len, target_len]
            colors = ['blue', 'red', 'green']
            labels = ['Original', 'Adversarial', 'Target'] if i == 0 else [None, None, None]
            
            for xi, yi, ci, li in zip(x, y, colors, labels):
                axes[2, 2].bar(xi, yi, width=0.4, color=ci, label=li, alpha=0.7)
        
        axes[2, 2].set_title('Transcription Lengths')
        axes[2, 2].set_xlabel('Samples')
        axes[2, 2].set_ylabel('Word Count')
        if results:
            axes[2, 2].legend()
        
        plt.tight_layout()
        plt.savefig('diagnostic_results.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Diagnostic plots saved to 'diagnostic_results.png'")
        
        return fig
    
    def save_results(self, results):
        """Save diagnostic results to JSON"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open('diagnostic_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Results saved to 'diagnostic_results.json'")

# Usage
if __name__ == "__main__":
    # Initialize diagnostics
    diagnostics = AttackDiagnostics(model, processor, device='cuda')
    
    # Select test samples (pick interesting cases)
    test_indices = [
        0,   # First sample
        100, # Sample that had finite loss
        500  # Sample that had infinite loss
    ]
    
    test_audio = [audio_samples[i] for i in test_indices]
    test_targets = [target_texts[i] for i in test_indices]
    
    # Run complete diagnostics
    results = diagnostics.run_complete_diagnostics(
        test_audio, 
        test_targets, 
        n_samples=3
    )
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUITE COMPLETE")
    print("=" * 60)
    print("Check 'diagnostic_results.png' for visualizations")
    print("Check 'diagnostic_results.json' for detailed data")