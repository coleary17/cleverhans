"""
CTC-based adversarial audio attack implementation.
Reproduces the original method using modern wav2vec2 CTC models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
import time
from typing import List, Tuple, Dict

from .ctc_audio_utils import (
    CTCASRModel, load_audio_file, save_audio_file, 
    audio_to_tensor, tensor_to_audio, parse_data_file
)
from .masking_threshold import generate_th, Transform


class CTCAdversarialAttack:
    """
    Two-stage adversarial attack against CTC-based ASR systems.
    
    Stage 1: Optimize adversarial perturbations to fool the CTC model
    Stage 2: Refine perturbations to be imperceptible using psychoacoustic masking
    """
    
    def __init__(self, 
                 model_size: str = 'base',
                 device: str = 'auto',
                 batch_size: int = 5,
                 window_size: int = 2048,
                 initial_bound: float = 0.15,
                 lr_stage1: float = 0.05,
                 lr_stage2: float = 0.005,
                 num_iter_stage1: int = 1000,
                 num_iter_stage2: int = 4000):
        
        # Auto-detect best available device
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
        
        # Initialize CTC ASR model
        self.asr_model = CTCASRModel(model_size=model_size, device=self.device)
        print(f"Model info: {self.asr_model.get_model_info()}")
        
        # Initialize transform for PSD computation (reuse from existing code)
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
            
            # Keep audio in float32 range for CTC model
            audio = audio.astype(np.float32)
            
            audios.append(audio)
            lengths.append(len(audio))
            
            # Compute masking threshold (reuse existing implementation)
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
    
    def stage1_attack(self, batch: Dict) -> torch.Tensor:
        """
        Stage 1: Optimize adversarial perturbations to fool the CTC model.
        
        Args:
            batch: Batch data dictionary
            
        Returns:
            Adversarial audio tensor
        """
        print("Starting CTC Stage 1 Attack...")
        
        audios = batch['audios']
        original_texts = batch['original_texts']
        target_texts = batch['target_texts']
        masks = batch['masks']
        lengths = batch['lengths']
        
        # Initialize adversarial perturbations
        delta = torch.zeros_like(audios, requires_grad=True, device=self.device)
        
        # Initialize rescale factors (match actual batch size)
        actual_batch_size = min(self.batch_size, audios.shape[0])
        rescale = torch.ones(actual_batch_size, 1, device=self.device)
        
        optimizer = optim.Adam([delta], lr=self.lr_stage1)
        
        # Initial transcriptions
        print("Initial transcriptions:")
        for i, (original_text, target_text) in enumerate(zip(original_texts, target_texts)):
            if i < audios.shape[0]:
                audio_np = audios[i, :lengths[i]].cpu().numpy()
                pred = self.asr_model.transcribe(audio_np)
                print(f"Example {i}:")
                print(f"  - Original: '{original_text}'")
                print(f"  - Target:   '{target_text}'")
                print(f"  - Predicted:'{pred}'")
        
        best_deltas = [None] * self.batch_size
        best_losses = [float('inf')] * self.batch_size
        
        for iteration in range(self.num_iter_stage1):
            optimizer.zero_grad()
            
            # Apply perturbations with bounds and rescaling
            bounded_delta = torch.clamp(delta, -self.initial_bound, self.initial_bound)
            scaled_delta = bounded_delta * rescale
            perturbed_audio = scaled_delta * masks + audios
            
            # Compute loss for each example in batch
            total_loss = 0
            individual_losses = []
            
            for i in range(min(self.batch_size, audios.shape[0])):
                try:
                    # Use CTC attack loss (negative CTC loss)
                    loss = self.asr_model.compute_attack_loss(
                        perturbed_audio[i, :lengths[i]], 
                        target_texts[i]
                    )
                    total_loss += loss
                    individual_losses.append(loss.item())
                    
                    # Check if attack succeeded and save best
                    if iteration % 10 == 0:
                        # Detach for transcription
                        audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                        pred = self.asr_model.transcribe(audio_sample_np)
                        
                        if pred.lower().strip() == target_texts[i].lower().strip():
                            print(f"SUCCESS at iteration {iteration} for example {i}")
                            print(f"  - Predicted: '{pred}'")
                            print(f"  - Target:    '{target_texts[i]}'")
                            
                            # Update rescale factor
                            current_max = torch.max(torch.abs(bounded_delta[i]))
                            if rescale[i] * self.initial_bound > current_max:
                                rescale[i] = current_max / self.initial_bound
                            rescale[i] *= 0.8
                            
                            # Save best example
                            best_deltas[i] = perturbed_audio[i].clone()
                            best_losses[i] = loss.item()
                            
                        elif loss.item() < best_losses[i]:
                            # Save best loss even if not successful
                            best_deltas[i] = perturbed_audio[i].clone()
                            best_losses[i] = loss.item()
                            
                except Exception as e:
                    print(f"Error processing example {i}: {e}")
                    continue
            
            grad_norm = 0
            delta_norm_before = delta.norm().item()

            if total_loss != 0 and torch.isfinite(total_loss):
                total_loss.backward()
                if delta.grad is not None:
                    grad_norm = delta.grad.norm().item()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                    
                    # CRITICAL: Use signed gradients in stage 1 (like original implementation)
                    # This is the key difference from the original method
                    delta.grad.sign_()
                optimizer.step()
            
            delta_norm_after = delta.norm().item()

            if iteration % 100 == 0:
                avg_loss = sum(individual_losses) / len(individual_losses) if individual_losses else 0
                print(f"Stage 1 - Iteration {iteration}, Avg Loss: {avg_loss:.4f}, Grad Norm: {grad_norm:.4f}, Delta Change: {delta_norm_after - delta_norm_before:.4f}")
                
                # Show current predictions
                for i in range(min(3, audios.shape[0])):  # Show first 3 examples
                    audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                    pred = self.asr_model.transcribe(audio_sample_np)
                    print(f"  Ex {i}: '{pred}' (target: '{target_texts[i]}')")
        
        # Use best deltas or final perturbed audio
        final_audio = torch.zeros_like(audios)
        for i in range(min(self.batch_size, audios.shape[0])):
            if best_deltas[i] is not None:
                final_audio[i] = best_deltas[i]
            else:
                # Fallback to final iteration
                bounded_delta = torch.clamp(delta, -self.initial_bound, self.initial_bound)
                scaled_delta = bounded_delta * rescale
                final_audio[i] = scaled_delta[i] * masks[i] + audios[i]
        
        print("CTC Stage 1 completed.")
        return final_audio
    
    def stage2_attack(self, batch: Dict, stage1_results: torch.Tensor) -> torch.Tensor:
        """
        Stage 2: Refine perturbations to be imperceptible using psychoacoustic masking.
        
        Args:
            batch: Batch data dictionary
            stage1_results: Results from stage 1
            
        Returns:
            Refined adversarial audio tensor
        """
        print("Starting CTC Stage 2 Attack (Psychoacoustic Masking)...")
        
        audios = batch['audios']
        target_texts = batch['target_texts']
        masks = batch['masks']
        lengths = batch['lengths']
        th_batch = batch['th_batch']
        psd_max_batch = batch['psd_max_batch']
        
        # Initialize delta from stage 1 results
        delta_init = stage1_results - audios
        delta = delta_init.clone().detach().requires_grad_(True)
        
        # Initialize alpha and rescale parameters
        alpha = torch.ones(self.batch_size, device=self.device) * 0.05
        rescale = torch.ones(self.batch_size, 1, device=self.device)
        
        optimizer = optim.Adam([delta], lr=self.lr_stage2)
        
        best_deltas = [None] * self.batch_size
        best_loss_th = [float('inf')] * self.batch_size
        
        for iteration in range(self.num_iter_stage2):
            optimizer.zero_grad()
            
            # Apply delta
            perturbed_audio = delta * masks + audios
            
            # Compute losses
            total_loss = 0
            
            for i in range(min(self.batch_size, audios.shape[0])):
                try:
                    # CTC loss component
                    ctc_loss = self.asr_model.compute_attack_loss(
                        perturbed_audio[i, :lengths[i]], 
                        target_texts[i]
                    )
                    
                    # Psychoacoustic masking loss
                    # Transform delta to frequency domain and compare with threshold
                    delta_sample = delta[i, :lengths[i]]
                    logits_delta = self.transform(delta_sample, psd_max_batch[i])
                    
                    # Masking threshold loss
                    th_tensor = torch.from_numpy(th_batch[i]).float().to(self.device)
                    loss_th = torch.mean(torch.relu(logits_delta - th_tensor))
                    
                    # Combined loss
                    combined_loss = ctc_loss + alpha[i] * loss_th
                    total_loss += combined_loss
                    
                    # Check success and update best
                    if iteration % 10 == 0:
                        audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                        pred = self.asr_model.transcribe(audio_sample_np)
                        
                        if (pred.lower().strip() == target_texts[i].lower().strip() and 
                            loss_th.item() < best_loss_th[i]):
                            
                            best_deltas[i] = perturbed_audio[i].clone()
                            best_loss_th[i] = loss_th.item()
                            print(f"Stage 2 SUCCESS for example {i} at iteration {iteration}")
                            print(f"  - Loss_th: {loss_th.item():.6f}")
                            
                        # Adjust alpha based on performance
                        if iteration % 20 == 0:
                            if pred.lower().strip() == target_texts[i].lower().strip():
                                alpha[i] *= 1.2  # Increase masking emphasis
                            else:
                                alpha[i] *= 0.8  # Decrease masking emphasis
                                alpha[i] = max(alpha[i], 0.0005)
                            
                except Exception as e:
                    print(f"Error in stage 2 for example {i}: {e}")
                    continue
            
            if total_loss != 0 and torch.is_finite(total_loss):
                total_loss.backward()
                optimizer.step()
            
            if iteration % 500 == 0:
                print(f"Stage 2 - Iteration {iteration}, Alpha values: {alpha.cpu().numpy()}")
                print(f"Best masking losses: {best_loss_th}")
        
        # Use best results or final iteration
        final_results = torch.zeros_like(audios)
        for i in range(min(self.batch_size, audios.shape[0])):
            if best_deltas[i] is not None:
                final_results[i] = best_deltas[i]
            else:
                final_results[i] = perturbed_audio[i]
        
        print("CTC Stage 2 completed.")
        return final_results
    
    def run_attack(self, data_file: str, root_dir: str = "./", output_dir: str = "./output_ctc", 
                   num_examples: int = 10, enable_stage2: bool = True):
        """
        Run the complete CTC adversarial attack.
        
        Args:
            data_file: Path to data file containing audio files and targets
            root_dir: Root directory for input audio files
            output_dir: Directory to save adversarial examples
            num_examples: Number of examples to process
            enable_stage2: Whether to run stage 2 (psychoacoustic masking)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Parse data file
        data = parse_data_file(data_file)
        
        # Limit to requested number of examples
        data = data[:num_examples]
        print(f"Processing {len(data)} examples with CTC model")
        
        # Process in batches
        for batch_idx in range(0, len(data), self.batch_size):
            batch_data = data[batch_idx:batch_idx + self.batch_size]
            audio_files = [item[0] for item in batch_data]
            original_texts = [item[1] for item in batch_data]
            target_texts = [item[2] for item in batch_data]
            
            print(f"\n=== Processing CTC batch {batch_idx // self.batch_size + 1} ===")
            print(f"Audio files: {audio_files}")
            print(f"Target texts: {target_texts}")
            
            try:
                # Prepare batch
                batch = self.prepare_batch(batch_data, root_dir)
                
                # Stage 1 attack
                stage1_results = self.stage1_attack(batch)
                
                # Save stage 1 results
                for i, audio_file in enumerate(audio_files):
                    if i < stage1_results.shape[0]:
                        name = Path(audio_file).stem
                        output_file = output_path / f"{name}_stage1_ctc.wav"
                        audio_data = stage1_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                        save_audio_file(audio_data, str(output_file), 16000)
                        
                        # Calculate distortion
                        original_audio = batch['audios'][i, :batch['lengths'][i]].cpu().numpy()
                        adversarial_audio = stage1_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                        distortion = np.max(np.abs(adversarial_audio - original_audio))
                        print(f"Stage 1 CTC distortion for {name}: {distortion:.2f}")
                        
                        # Test final transcription
                        final_pred = self.asr_model.transcribe(audio_data)
                        print(f"Final stage 1 prediction for {name}: '{final_pred}'")
                
                # Stage 2 attack (if enabled)
                if enable_stage2:
                    stage2_results = self.stage2_attack(batch, stage1_results)
                    
                    # Save stage 2 results
                    for i, audio_file in enumerate(audio_files):
                        if i < stage2_results.shape[0]:
                            name = Path(audio_file).stem
                            output_file = output_path / f"{name}_stage2_ctc.wav"
                            audio_data = stage2_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                            save_audio_file(audio_data, str(output_file), 16000)
                            
                            # Calculate distortion
                            original_audio = batch['audios'][i, :batch['lengths'][i]].cpu().numpy()
                            adversarial_audio = stage2_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                            distortion = np.max(np.abs(adversarial_audio - original_audio))
                            print(f"Stage 2 CTC distortion for {name}: {distortion:.2f}")
                            
                            # Test final transcription
                            final_pred = self.asr_model.transcribe(audio_data)
                            print(f"Final stage 2 prediction for {name}: '{final_pred}'")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx // self.batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nCTC Attack completed. Results saved to {output_dir}")


def main():
    """Main function to run CTC adversarial attack."""
    parser = argparse.ArgumentParser(description='Run CTC adversarial audio attack')
    parser.add_argument('--data_file', default='../adversarial_asr/read_data.txt', 
                       help='Path to data file')
    parser.add_argument('--root_dir', default='../adversarial_asr/', 
                       help='Root directory for audio files')
    parser.add_argument('--output_dir', default='./output_ctc', 
                       help='Output directory for adversarial examples')
    parser.add_argument('--model_size', default='base', 
                       choices=['base', 'large', 'large-lv60'],
                       help='CTC model size')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device to use (cpu/cuda/mps/auto, auto will detect best available)')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size for processing')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of examples to process')
    parser.add_argument('--lr_stage1', type=float, default=100.0,
                       help='Learning rate for stage 1')
    parser.add_argument('--lr_stage2', type=float, default=1.0,
                       help='Learning rate for stage 2')
    parser.add_argument('--num_iter_stage1', type=int, default=1000,
                       help='Number of iterations for stage 1')
    parser.add_argument('--num_iter_stage2', type=int, default=4000,
                       help='Number of iterations for stage 2')
    parser.add_argument('--disable_stage2', action='store_true',
                       help='Disable stage 2 (psychoacoustic masking)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CTC ADVERSARIAL ATTACK")
    print("=" * 60)
    print(f"Model size: {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Examples: {args.num_examples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Stage 2 enabled: {not args.disable_stage2}")
    print("=" * 60)
    
    # Initialize attack
    attack = CTCAdversarialAttack(
        model_size=args.model_size,
        device=args.device,
        batch_size=args.batch_size,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        num_iter_stage1=args.num_iter_stage1,
        num_iter_stage2=args.num_iter_stage2
    )
    
    # Run attack
    attack.run_attack(
        data_file=args.data_file, 
        root_dir=args.root_dir, 
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        enable_stage2=not args.disable_stage2
    )


if __name__ == "__main__":
    main()
