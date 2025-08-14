#!/usr/bin/env python3
"""
Diagnostic script to analyze audio distortion in CTC adversarial attacks.
Tests different parameter configurations on a single audio example with detailed monitoring.
"""

import torch
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
import time
import json
from typing import List, Dict, Tuple

from src.adversarial_asr_modern.ctc_audio_utils import (
    CTCASRModel, load_audio_file, save_audio_file, 
    audio_to_tensor, tensor_to_audio, parse_data_file
)


class DistortionDiagnostic:
    """
    Diagnostic tool for analyzing adversarial attack distortion patterns.
    """
    
    def __init__(self, model_size: str = 'base', device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing diagnostic on device: {self.device}")
        
        # Initialize CTC ASR model
        self.asr_model = CTCASRModel(model_size=model_size, device=self.device)
        print(f"Model info: {self.asr_model.get_model_info()}")
    
    def calculate_distortion_metrics(self, original: np.ndarray, perturbed: np.ndarray) -> Dict:
        """Calculate comprehensive distortion metrics."""
        if len(original) != len(perturbed):
            min_len = min(len(original), len(perturbed))
            original = original[:min_len]
            perturbed = perturbed[:min_len]
        
        # Calculate different distortion metrics
        l_inf = np.max(np.abs(perturbed - original))
        l2 = np.sqrt(np.mean((perturbed - original) ** 2))
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((perturbed - original) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Audio range analysis
        original_range = (np.min(original), np.max(original))
        perturbed_range = (np.min(perturbed), np.max(perturbed))
        clipping = np.sum(np.abs(perturbed) > 1.0)
        
        return {
            'l_inf': float(l_inf),
            'l2': float(l2),
            'snr_db': float(snr_db),
            'original_range': original_range,
            'perturbed_range': perturbed_range,
            'clipping_samples': int(clipping),
            'clipping_percentage': float(clipping / len(perturbed) * 100)
        }
    
    def run_single_config_test(self, 
                              original_audio: np.ndarray,
                              target_text: str,
                              lr: float,
                              bound: float,
                              num_iterations: int,
                              checkpoint_intervals: List[int],
                              output_dir: Path) -> Dict:
        """
        Run adversarial attack with specific configuration and save checkpoints.
        """
        print(f"\n=== Testing lr={lr}, bound={bound} ===")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(original_audio.astype(np.float32)).to(self.device)
        delta = torch.zeros_like(audio_tensor, requires_grad=True, device=self.device)
        
        optimizer = optim.Adam([delta], lr=lr)
        
        # Results tracking
        results = {
            'config': {'lr': lr, 'bound': bound},
            'checkpoints': {},
            'metrics_timeline': [],
            'transcription_timeline': [],
            'loss_timeline': []
        }
        
        # Get original transcription
        original_transcription = self.asr_model.transcribe(original_audio)
        print(f"Original transcription: '{original_transcription}'")
        print(f"Target text: '{target_text}'")
        
        # Save original audio
        save_audio_file(original_audio, str(output_dir / "original.wav"), 16000)
        
        best_loss = float('inf')
        best_transcription = ""
        
        for iteration in range(num_iterations + 1):
            if iteration > 0:  # Skip optimization on iteration 0 (original)
                optimizer.zero_grad()
                
                # Apply perturbations with bounds
                bounded_delta = torch.clamp(delta, -bound, bound)
                perturbed_audio = audio_tensor + bounded_delta
                
                # Compute loss
                try:
                    loss = self.asr_model.compute_attack_loss(perturbed_audio, target_text)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        
                        # Apply signed gradients (like original implementation)
                        if delta.grad is not None:
                            grad_norm = delta.grad.norm().item()
                            delta.grad.sign_()
                        else:
                            grad_norm = 0.0
                        
                        optimizer.step()
                        
                        # Track best loss
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                    else:
                        loss = torch.tensor(float('inf'))
                        grad_norm = 0.0
                        
                except Exception as e:
                    print(f"Error at iteration {iteration}: {e}")
                    loss = torch.tensor(float('inf'))
                    grad_norm = 0.0
            else:
                # Iteration 0: original audio
                bounded_delta = torch.zeros_like(delta)
                perturbed_audio = audio_tensor
                loss = torch.tensor(0.0)
                grad_norm = 0.0
            
            # Check if this is a checkpoint iteration
            if iteration in checkpoint_intervals or iteration == 0:
                print(f"  Checkpoint at iteration {iteration}")
                
                # Get current perturbed audio
                current_audio = perturbed_audio.detach().cpu().numpy()
                
                # Calculate distortion metrics
                distortion_metrics = self.calculate_distortion_metrics(original_audio, current_audio)
                
                # Get transcription
                try:
                    current_transcription = self.asr_model.transcribe(current_audio)
                except:
                    current_transcription = "[ERROR]"
                
                # Save audio checkpoint
                checkpoint_path = output_dir / f"iter_{iteration:04d}.wav"
                
                # Ensure audio is in valid range before saving
                clipped_audio = np.clip(current_audio, -1.0, 1.0)
                save_audio_file(clipped_audio, str(checkpoint_path), 16000)
                
                # Store checkpoint data
                checkpoint_data = {
                    'iteration': iteration,
                    'loss': float(loss.item()) if torch.isfinite(loss) else float('inf'),
                    'grad_norm': grad_norm,
                    'transcription': current_transcription,
                    'transcription_length': len(current_transcription),
                    'distortion_metrics': distortion_metrics,
                    'audio_file': str(checkpoint_path.name)
                }
                
                results['checkpoints'][iteration] = checkpoint_data
                
                # Print status
                print(f"    Loss: {checkpoint_data['loss']:.4f}")
                print(f"    Transcription ({len(current_transcription)}): '{current_transcription}'")
                print(f"    L∞ distortion: {distortion_metrics['l_inf']:.6f}")
                print(f"    SNR: {distortion_metrics['snr_db']:.2f} dB")
                print(f"    Clipping: {distortion_metrics['clipping_percentage']:.1f}%")
                
                if current_transcription:
                    best_transcription = current_transcription
            
            # Track timeline data every 50 iterations
            if iteration % 50 == 0:
                current_audio = perturbed_audio.detach().cpu().numpy()
                try:
                    current_transcription = self.asr_model.transcribe(current_audio)
                except:
                    current_transcription = "[ERROR]"
                
                results['metrics_timeline'].append({
                    'iteration': iteration,
                    'loss': float(loss.item()) if torch.isfinite(loss) else float('inf'),
                    'l_inf': float(torch.max(torch.abs(bounded_delta)).item())
                })
                results['transcription_timeline'].append({
                    'iteration': iteration,
                    'transcription': current_transcription,
                    'length': len(current_transcription)
                })
        
        # Final summary
        results['summary'] = {
            'original_transcription': original_transcription,
            'target_text': target_text,
            'best_loss': best_loss,
            'best_transcription': best_transcription,
            'final_transcription': results['transcription_timeline'][-1]['transcription'] if results['transcription_timeline'] else ""
        }
        
        print(f"Final transcription: '{results['summary']['final_transcription']}'")
        print(f"Best loss achieved: {best_loss:.4f}")
        
        return results
    
    def run_diagnostic(self, 
                      audio_file: str, 
                      original_text: str,
                      target_text: str,
                      output_dir: str = "./diagnostic_output",
                      configurations: List[Dict] = None):
        """
        Run comprehensive diagnostic tests with multiple configurations.
        """
        if configurations is None:
            configurations = [
                {'lr': 100.0, 'bound': 0.1, 'name': 'current_high'},
                {'lr': 10.0, 'bound': 0.05, 'name': 'moderate'},
                {'lr': 1.0, 'bound': 0.01, 'name': 'conservative'},
                {'lr': 0.1, 'bound': 0.005, 'name': 'ultra_conservative'}
            ]
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load audio
        print(f"Loading audio file: {audio_file}")
        audio, sr = load_audio_file(audio_file, target_sr=16000)
        print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
        print(f"Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
        
        # Save original audio to output directory
        save_audio_file(audio, str(output_path / "original_reference.wav"), 16000)
        
        # Get original transcription
        original_transcription = self.asr_model.transcribe(audio)
        print(f"Original transcription: '{original_transcription}'")
        print(f"Target text: '{target_text}'")
        
        # Define checkpoint intervals
        checkpoint_intervals = [0, 100, 250, 500, 750, 1000]
        
        # Run tests for each configuration
        all_results = {}
        
        for config in configurations:
            config_name = config['name']
            config_dir = output_path / config_name
            config_dir.mkdir(exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"TESTING CONFIGURATION: {config_name}")
            print(f"Learning Rate: {config['lr']}, Bound: {config['bound']}")
            print(f"{'='*60}")
            
            try:
                results = self.run_single_config_test(
                    original_audio=audio,
                    target_text=target_text,
                    lr=config['lr'],
                    bound=config['bound'],
                    num_iterations=1000,
                    checkpoint_intervals=checkpoint_intervals,
                    output_dir=config_dir
                )
                
                all_results[config_name] = results
                
                # Save individual config results
                with open(config_dir / "results.json", 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Create config summary
                self.create_config_summary(results, config_dir / "summary.txt")
                
            except Exception as e:
                print(f"ERROR in configuration {config_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create overall comparison report
        self.create_comparison_report(all_results, output_path / "comparison_report.txt")
        
        # Save complete results
        with open(output_path / "complete_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("DIAGNOSTIC COMPLETE")
        print(f"Results saved to: {output_path}")
        print("Check the audio files and comparison report for analysis.")
        print(f"{'='*60}")
        
        return all_results
    
    def create_config_summary(self, results: Dict, output_file: Path):
        """Create a human-readable summary for a single configuration."""
        with open(output_file, 'w') as f:
            f.write("CONFIGURATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            config = results['config']
            f.write(f"Learning Rate: {config['lr']}\n")
            f.write(f"Perturbation Bound: {config['bound']}\n\n")
            
            summary = results['summary']
            f.write(f"Original Transcription: '{summary['original_transcription']}'\n")
            f.write(f"Target Text: '{summary['target_text']}'\n")
            f.write(f"Final Transcription: '{summary['final_transcription']}'\n")
            f.write(f"Best Loss: {summary['best_loss']:.4f}\n\n")
            
            f.write("CHECKPOINT ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for iteration, data in sorted(results['checkpoints'].items()):
                f.write(f"\nIteration {iteration}:\n")
                f.write(f"  Transcription ({data['transcription_length']}): '{data['transcription']}'\n")
                f.write(f"  Loss: {data['loss']:.4f}\n")
                
                metrics = data['distortion_metrics']
                f.write(f"  L∞ Distortion: {metrics['l_inf']:.6f}\n")
                f.write(f"  L2 Distortion: {metrics['l2']:.6f}\n")
                f.write(f"  SNR: {metrics['snr_db']:.2f} dB\n")
                f.write(f"  Audio Range: [{metrics['perturbed_range'][0]:.3f}, {metrics['perturbed_range'][1]:.3f}]\n")
                f.write(f"  Clipping: {metrics['clipping_percentage']:.1f}%\n")
    
    def create_comparison_report(self, all_results: Dict, output_file: Path):
        """Create a comparison report across all configurations."""
        with open(output_file, 'w') as f:
            f.write("CONFIGURATION COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary table
            f.write("FINAL RESULTS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Config':<20} {'LR':<8} {'Bound':<8} {'Final Length':<12} {'Best Loss':<10} {'Final Transcription'}\n")
            f.write("-" * 100 + "\n")
            
            for config_name, results in all_results.items():
                config = results['config']
                summary = results['summary']
                final_trans = summary['final_transcription']
                final_length = len(final_trans)
                
                f.write(f"{config_name:<20} {config['lr']:<8} {config['bound']:<8} {final_length:<12} {summary['best_loss']:<10.4f} '{final_trans}'\n")
            
            f.write("\n\nDETAILED ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for config_name, results in all_results.items():
                f.write(f"\n{config_name.upper()}:\n")
                
                # Find the checkpoint with the longest transcription
                best_checkpoint = None
                max_length = 0
                
                for iteration, data in results['checkpoints'].items():
                    if data['transcription_length'] > max_length:
                        max_length = data['transcription_length']
                        best_checkpoint = data
                
                if best_checkpoint:
                    f.write(f"  Best transcription length: {max_length} characters\n")
                    f.write(f"  Best transcription: '{best_checkpoint['transcription']}'\n")
                    f.write(f"  At iteration: {best_checkpoint['iteration']}\n")
                    f.write(f"  Distortion at best: L∞={best_checkpoint['distortion_metrics']['l_inf']:.6f}\n")
                    f.write(f"  SNR at best: {best_checkpoint['distortion_metrics']['snr_db']:.2f} dB\n")
                
                # Transcription evolution
                f.write(f"  Transcription evolution:\n")
                for timeline_entry in results['transcription_timeline'][::2]:  # Every other entry
                    iter_num = timeline_entry['iteration']
                    trans = timeline_entry['transcription']
                    length = timeline_entry['length']
                    f.write(f"    Iter {iter_num:4d}: ({length:2d}) '{trans}'\n")
            
            f.write("\n\nRECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            # Find configuration with best balance
            best_config = None
            best_score = 0
            
            for config_name, results in all_results.items():
                # Score based on transcription length and reasonable distortion
                summary = results['summary']
                final_length = len(summary['final_transcription'])
                
                # Find lowest distortion checkpoint with decent transcription
                reasonable_distortion = False
                for data in results['checkpoints'].values():
                    if data['distortion_metrics']['l_inf'] < 0.1 and data['transcription_length'] > 5:
                        reasonable_distortion = True
                        break
                
                score = final_length * (2 if reasonable_distortion else 1)
                if score > best_score:
                    best_score = score
                    best_config = config_name
            
            if best_config:
                f.write(f"Best performing configuration: {best_config}\n")
                f.write(f"Produces the longest coherent transcriptions with reasonable distortion.\n")
            
            f.write("\nNext steps:\n")
            f.write("1. Listen to the audio files to confirm quality\n")
            f.write("2. Focus on configurations that maintain transcription length > 10 characters\n")
            f.write("3. Consider hybrid approaches or further parameter tuning\n")


def main():
    """Main function for running diagnostic tests."""
    parser = argparse.ArgumentParser(description='Diagnose audio distortion in CTC adversarial attacks')
    parser.add_argument('--audio_file', 
                       default='../adversarial_asr/LibriSpeech/test-clean/61/70968/61-70968-0011.wav',
                       help='Path to audio file to test')
    parser.add_argument('--original_text', 
                       default='SIX SPOONS OF FRESH SNOW PEAS FIVE THICK SLABS OF BLUE CHEESE AND MAYBE A SNACK FOR HER BROTHER BOB',
                       help='Original transcription text')
    parser.add_argument('--target_text', 
                       default='hello world this is a test',
                       help='Target text for adversarial attack')
    parser.add_argument('--output_dir', 
                       default='./diagnostic_output',
                       help='Output directory for diagnostic results')
    parser.add_argument('--model_size', 
                       default='base',
                       choices=['base', 'large', 'large-lv60'],
                       help='CTC model size')
    parser.add_argument('--device', 
                       default='cpu',
                       choices=['cpu', 'cuda'], 
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CTC ADVERSARIAL ATTACK - DISTORTION DIAGNOSTIC")
    print("=" * 60)
    print(f"Audio file: {args.audio_file}")
    print(f"Original text: {args.original_text}")
    print(f"Target text: {args.target_text}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_size}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Initialize diagnostic
    diagnostic = DistortionDiagnostic(model_size=args.model_size, device=args.device)
    
    # Run diagnostic
    results = diagnostic.run_diagnostic(
        audio_file=args.audio_file,
        original_text=args.original_text,
        target_text=args.target_text,
        output_dir=args.output_dir
    )
    
    print(f"\nDiagnostic completed successfully!")
    print(f"Check {args.output_dir} for detailed results and audio files.")


if __name__ == "__main__":
    main()
