"""
Modernized adversarial audio attack using PyTorch and OpenAI Whisper.
Ported from the original TensorFlow 1.x implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
import time
from typing import List, Tuple, Dict
import pandas as pd
import json
from datetime import datetime

from .audio_utils import (
    WhisperASRModel, load_audio_file, save_audio_file, 
    audio_to_tensor, tensor_to_audio, parse_data_file
)
from .masking_threshold import generate_th, Transform


class AdversarialAttack:
    """
    Two-stage adversarial attack against ASR systems.
    
    Stage 1: Optimize adversarial perturbations to fool the ASR model
    Stage 2: Refine perturbations to be imperceptible using psychoacoustic masking
    """
    
    def __init__(self, 
                 model_name: str = "openai/whisper-base",
                 device: str = 'auto',
                 batch_size: int = 5,
                 window_size: int = 2048,
                 initial_bound: float = 0.1,
                 lr_stage1: float = 0.05,
                 lr_stage2: float = 0.005,
                 num_iter_stage1: int = 1000,
                 num_iter_stage2: int = 4000,
                 log_interval: int = 10,
                 verbose: bool = False,
                 save_audio: bool = False):
        
        # Auto-detect best available device
        print(f"Using device: {device}")
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Auto-detected CUDA GPU available, using: {self.device}")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
                print(f"Auto-detected Apple Silicon GPU (MPS), using: {self.device}")
            else:
                self.device = torch.device('cpu')
                print(f"No GPU detected, using: {self.device}")
        elif device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, falling back to CPU")
            self.device = torch.device('cpu')
        elif device == 'mps':
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
                print(f"Using Apple Silicon GPU (MPS): {self.device}")
            else:
                print(f"Warning: MPS requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            print(f"Using specified device: {self.device}")
        
        self.batch_size = batch_size
        self.window_size = window_size
        self.initial_bound = initial_bound
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage2 = num_iter_stage2
        self.log_interval = log_interval
        self.verbose = verbose
        self.save_audio = save_audio
        
        # Initialize ASR model
        self.asr_model = WhisperASRModel(model_name, device=self.device)
        
        # Initialize transform for PSD computation
        self.transform = Transform(window_size, device=self.device)
        
    def prepare_batch(self, batch_data: List[Tuple[str, str, str]], root_dir: str = "./") -> Dict:
        """
        Prepare batch of audio files and compute masking thresholds.
        
        Args:
            batch_data: List of tuples (audio_file, original_text, target_text)
            root_dir: Root directory for audio files
            
        Returns:
            Dictionary containing batch data
        """
        self.prepare_batch_data = batch_data
        audio_files = [item[0] for item in batch_data]
        target_texts = [item[2] for item in batch_data]

        audios = []
        lengths = []
        th_batch = []
        psd_max_batch = []
        
        # Load audio files
        for audio_file in audio_files:
            audio_path = Path(root_dir) / audio_file
            audio, sr = load_audio_file(str(audio_path), target_sr=16000)
            
            # Whisper expects audio in [-1, 1] range
            audio = audio.astype(np.float32)
            # Normalize if needed
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            elif max_val < 0.1:
                # Scale up very quiet audio
                audio = audio / max_val * 0.5
                
            audios.append(audio)
            lengths.append(len(audio))
            
            # Compute masking threshold
            th, psd_max = generate_th(audio, 16000, self.window_size)
            th_batch.append(th)
            psd_max_batch.append(psd_max)
        
        # Pad to max length
        max_length = max(lengths)
        padded_audios = np.zeros((len(audios), max_length))
        masks = np.zeros((len(audios), max_length))
        
        for i, (audio, length) in enumerate(zip(audios, lengths)):
            padded_audios[i, :length] = audio
            masks[i, :length] = 1
        
        return {
            'audios': torch.from_numpy(padded_audios).float().to(self.device),
            'original_texts': [item[1] for item in self.prepare_batch_data],
            'target_texts': target_texts,
            'th_batch': th_batch,
            'psd_max_batch': np.array(psd_max_batch),
            'masks': torch.from_numpy(masks).float().to(self.device),
            'lengths': lengths,
            'max_length': max_length
        }
    
    def stage1_attack(self, batch: Dict) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Stage 1: Optimize adversarial perturbations to fool the ASR model.
        
        Args:
            batch: Batch data dictionary
            
        Returns:
            Adversarial audio tensor
        """
        print("=" * 60)
        print("Starting Stage 1 Attack...")
        print(f"Logging predictions every {self.log_interval} iterations")
        print(f"Verbose mode: {self.verbose}")
        print("=" * 60)
        
        audios = batch['audios']
        original_texts = batch['original_texts']
        target_texts = batch['target_texts']
        masks = batch['masks']
        lengths = batch['lengths']
        
        # Initialize adversarial perturbations
        delta = torch.zeros_like(audios, requires_grad=True, device=self.device)
        
        # Initialize rescale factors
        rescale = torch.ones(self.batch_size, 1, device=self.device)
        
        optimizer = optim.Adam([delta], lr=self.lr_stage1)
        
        # Track success iterations for each example
        success_iterations = [-1] * self.batch_size
        best_deltas = [None] * self.batch_size
        attack_results = []
        
        # Initial transcriptions
        print("\n[INITIAL STATE]")
        for i, (original_text, target_text) in enumerate(zip(original_texts, target_texts)):
            if i < audios.shape[0]:
                audio_np = audios[i, :lengths[i]].cpu().numpy()
                pred = self.asr_model.transcribe(audio_np)
                print(f"Example {i}:")
                print(f"  Original: '{original_text}'")
                print(f"  Target:   '{target_text}'")
                print(f"  Current:  '{pred}'")
                print()
        
        # Main optimization loop
        for iteration in range(self.num_iter_stage1):
            optimizer.zero_grad()
            
            # Apply perturbations with bounds and rescaling
            bounded_delta = torch.clamp(delta, -self.initial_bound, self.initial_bound)
            scaled_delta = bounded_delta * rescale
            perturbed_audio = (scaled_delta * masks + audios).clamp(-1.0, 1.0)
            
            # Compute loss for each example in batch
            total_loss = 0
            individual_losses = []
            
            for i in range(min(self.batch_size, audios.shape[0])):
                try:
                    # Loss must be computed on the tensor with gradients
                    loss = self.asr_model.compute_loss(perturbed_audio[i, :lengths[i]], target_texts[i])
                    total_loss += loss
                    individual_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"Error processing example {i}: {e}")
                    individual_losses.append(float('inf'))
                    continue
            
            if total_loss > 0:
                total_loss.backward()
                
                # CRITICAL: Use signed gradients like the original paper
                # This is key to making the attack work!
                if delta.grad is not None:
                    # Option 1: Sign the gradients (original method)
                    # delta.grad.sign_()
                    
                    # Option 2: Clip gradients
                    torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                
                optimizer.step()
            
            # Log predictions at specified intervals
            should_log = (iteration % self.log_interval == 0) or (iteration == self.num_iter_stage1 - 1)
            
            if should_log or self.verbose:
                print(f"\n[Iteration {iteration}/{self.num_iter_stage1}]")
                print(f"Total Loss: {total_loss:.4f}")
                
                # Check predictions for all examples
                for i in range(min(self.batch_size, audios.shape[0])):
                    if success_iterations[i] != -1:
                        # Already succeeded, skip detailed logging
                        if self.verbose:
                            print(f"Example {i}: SUCCESS (achieved at iteration {success_iterations[i]})")
                        continue
                    
                    try:
                        # Get current prediction
                        audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                        pred = self.asr_model.transcribe(audio_sample_np)
                        
                        # Calculate perturbation stats
                        perturbation = (perturbed_audio[i, :lengths[i]] - audios[i, :lengths[i]]).detach()
                        max_pert = torch.max(torch.abs(perturbation)).item()
                        mean_pert = torch.mean(torch.abs(perturbation)).item()
                        
                        # Check for success
                        success = pred.lower().strip() == target_texts[i].lower().strip()
                        
                        if success and success_iterations[i] == -1:
                            success_iterations[i] = iteration
                            print(f"Example {i}: ✓ SUCCESS!")
                            print(f"  Target:   '{target_texts[i]}'")
                            print(f"  Achieved: '{pred}'")
                            print(f"  Loss: {individual_losses[i]:.4f}")
                            print(f"  Max perturbation: {max_pert:.2f}, Mean: {mean_pert:.2f}")
                            
                            # Update rescale factor
                            current_max = torch.max(torch.abs(bounded_delta[i]))
                            if rescale[i] * self.initial_bound > current_max:
                                rescale[i] = current_max / self.initial_bound
                            rescale[i] *= 0.8
                            
                            # Save best example
                            best_deltas[i] = perturbed_audio[i].clone()
                        else:
                            print(f"Example {i}:")
                            print(f"  Target:   '{target_texts[i]}'")
                            print(f"  Current:  '{pred}'")
                            print(f"  Loss: {individual_losses[i]:.4f}")
                            if self.verbose:
                                print(f"  Max pert: {max_pert:.2f}, Mean: {mean_pert:.2f}")
                            
                    except Exception as e:
                        print(f"Example {i}: Error - {e}")
                
                # Summary statistics
                successful = sum(1 for s in success_iterations if s != -1)
                print(f"\nProgress: {successful}/{min(self.batch_size, audios.shape[0])} successful")
                print("-" * 40)
        
        # Final summary and collect results
        print("\n" + "=" * 60)
        print("STAGE 1 COMPLETE - FINAL RESULTS:")
        
        # Use best deltas or final perturbed audio
        final_audio = torch.zeros_like(audios)
        bounded_delta = torch.clamp(delta, -self.initial_bound, self.initial_bound)
        scaled_delta = bounded_delta * rescale
        
        for i in range(min(self.batch_size, audios.shape[0])):
            # Get final audio
            if best_deltas[i] is not None:
                final_audio[i] = best_deltas[i]
            else:
                # Fallback to final iteration
                final_audio[i] = (scaled_delta[i] * masks[i] + audios[i]).clamp(-1.0, 1.0)
            
            # Get final prediction
            audio_sample_np = final_audio[i, :lengths[i]].detach().cpu().numpy()
            final_pred = self.asr_model.transcribe(audio_sample_np)
            
            # Calculate final perturbation stats
            perturbation = (final_audio[i, :lengths[i]] - audios[i, :lengths[i]]).detach()
            max_pert = torch.max(torch.abs(perturbation)).item()
            mean_pert = torch.mean(torch.abs(perturbation)).item()
            
            # Get final loss
            try:
                with torch.no_grad():
                    loss_val = self.asr_model.compute_loss(final_audio[i:i+1], target_texts[i]).item()
            except:
                loss_val = -1
            
            # Create result entry
            result = {
                'example_idx': i,
                'original_text': original_texts[i],
                'target_text': target_texts[i],
                'final_text': final_pred,
                'success': success_iterations[i] != -1,
                'success_iteration': success_iterations[i] if success_iterations[i] != -1 else self.num_iter_stage1,
                'final_loss': loss_val,
                'max_perturbation': max_pert,
                'mean_perturbation': mean_pert,
                'stage': 'stage1'
            }
            attack_results.append(result)
            
            # Print summary
            if success_iterations[i] != -1:
                print(f"Example {i}: SUCCESS at iteration {success_iterations[i]}")
            else:
                print(f"Example {i}: FAILED")
                print(f"  Target: '{target_texts[i]}'")
                print(f"  Final:  '{final_pred}'")
        
        print("=" * 60)
        
        return final_audio, attack_results
    
    def run_attack(self, data_file: str, root_dir: str = "./", output_dir: str = "./output", 
                   results_file: str = None):
        """
        Run the complete adversarial attack.
        
        Args:
            data_file: Path to data file containing audio files and targets
            root_dir: Root directory for input audio files
            output_dir: Directory to save adversarial examples (only if save_audio=True)
            results_file: Path to save results CSV/JSON file
        """
        output_path = Path(output_dir)
        if self.save_audio:
            output_path.mkdir(exist_ok=True)
        
        # Initialize results collection
        all_results = []
        
        # Parse data file
        data = parse_data_file(data_file)
        
        # Process in batches
        for batch_idx in range(0, len(data), self.batch_size):
            batch_data = data[batch_idx:batch_idx + self.batch_size]
            audio_files = [item[0] for item in batch_data]
            original_texts = [item[1] for item in batch_data]
            target_texts = [item[2] for item in batch_data]
            
            print(f"\nProcessing batch {batch_idx // self.batch_size + 1}")
            print(f"Audio files: {audio_files}")
            print(f"Target texts: {target_texts}")
            
            try:
                # Prepare batch
                batch = self.prepare_batch(batch_data, root_dir)
                
                # Stage 1 attack
                stage1_results, batch_results = self.stage1_attack(batch)
                
                # Add file information to results
                for i, audio_file in enumerate(audio_files):
                    if i < len(batch_results):
                        name = Path(audio_file).stem
                        batch_results[i]['audio_file'] = audio_file
                        batch_results[i]['audio_name'] = name
                        
                        # Calculate distortion
                        if i < stage1_results.shape[0]:
                            original_audio = batch['audios'][i, :batch['lengths'][i]].cpu().numpy()
                            adversarial_audio = stage1_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                            distortion = np.max(np.abs(adversarial_audio - original_audio))
                            batch_results[i]['stage1_distortion'] = distortion
                            print(f"Stage 1 distortion for {name}: {distortion:.2f}")
                            
                            # Save audio if requested
                            if self.save_audio:
                                output_file = output_path / f"{name}_stage1.wav"
                                audio_data = stage1_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                                save_audio_file(audio_data, str(output_file), 16000)
                
                # Collect results
                all_results.extend(batch_results)
                
            except Exception as e:
                print(f"Error processing batch {batch_idx // self.batch_size + 1}: {e}")
                # Add failed entries for this batch
                for i, audio_file in enumerate(audio_files):
                    all_results.append({
                        'audio_file': audio_file,
                        'audio_name': Path(audio_file).stem,
                        'original_text': original_texts[i] if i < len(original_texts) else '',
                        'target_text': target_texts[i] if i < len(target_texts) else '',
                        'final_text': '',
                        'success': False,
                        'error': str(e),
                        'stage': 'failed'
                    })
                continue
        
        # Save results to file
        if results_file:
            self.save_results(all_results, results_file)
        else:
            # Default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"attack_results_{timestamp}.csv"
            self.save_results(all_results, results_file)
        
        # Print summary
        self.print_summary(all_results)
    
    def save_results(self, results: List[Dict], filepath: str):
        """Save attack results to CSV or JSON file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filepath} (JSON format)")
        else:
            # Default to CSV
            df = pd.DataFrame(results)
            df.to_csv(filepath, index=False)
            print(f"\nResults saved to {filepath} (CSV format)")
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics of attack results."""
        if not results:
            print("\nNo results to summarize")
            return
        
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = sum(1 for r in results if r.get('stage') == 'failed')
        
        print("\n" + "=" * 60)
        print("ATTACK SUMMARY")
        print("=" * 60)
        print(f"Total examples: {total}")
        print(f"Successful: {successful} ({100*successful/total:.1f}%)")
        print(f"Failed: {total - successful} ({100*(total-successful)/total:.1f}%)")
        
        if successful > 0:
            success_results = [r for r in results if r.get('success', False)]
            avg_iterations = np.mean([r['success_iteration'] for r in success_results])
            avg_loss = np.mean([r['final_loss'] for r in success_results if r.get('final_loss', -1) > 0])
            avg_max_pert = np.mean([r['max_perturbation'] for r in success_results if 'max_perturbation' in r])
            
            print(f"\nFor successful attacks:")
            print(f"  Average iterations: {avg_iterations:.1f}")
            print(f"  Average final loss: {avg_loss:.4f}")
            print(f"  Average max perturbation: {avg_max_pert:.4f}")
        
        if failed > 0:
            print(f"\nBatch processing errors: {failed}")
        
        print("=" * 60)


def main():
    """Main function to run adversarial attack."""
    parser = argparse.ArgumentParser(description='Run adversarial audio attack')
    parser.add_argument('--data_file', default='../adversarial_asr/read_data.txt', 
                       help='Path to data file')
    parser.add_argument('--root_dir', default='../adversarial_asr/', 
                       help='Root directory for audio files')
    parser.add_argument('--output_dir', default='./output', 
                       help='Output directory for adversarial examples')
    parser.add_argument('--model_name', default='openai/whisper-base',
                       help='Whisper model name')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device to use (cpu/cuda/mps/auto, auto will detect best available)') 
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size for processing')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Interval for logging predictions during optimization')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--save_audio', action='store_true',
                       help='Save adversarial audio files (default: False)')
    parser.add_argument('--results_file', default=None,
                       help='Path to save results CSV/JSON (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Initialize attack
    attack = AdversarialAttack(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        verbose=args.verbose,
        save_audio=args.save_audio
    )
    
    # Run attack
    attack.run_attack(args.data_file, args.root_dir, args.output_dir, args.results_file)


if __name__ == "__main__":
    main()
