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
                 device: str = 'cpu',
                 batch_size: int = 5,
                 window_size: int = 2048,
                 initial_bound: float = 2000.0,
                 lr_stage1: float = 100.0,
                 lr_stage2: float = 1.0,
                 num_iter_stage1: int = 1000,
                 num_iter_stage2: int = 4000):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.batch_size = batch_size
        self.window_size = window_size
        self.initial_bound = initial_bound
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage2 = num_iter_stage2
        
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
            
            # Keep audio in normalized range for Whisper
            audio = audio.astype(np.float32)
            # Ensure audio is properly normalized
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
                
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
    
    def stage1_attack(self, batch: Dict) -> torch.Tensor:
        """
        Stage 1: Optimize adversarial perturbations to fool the ASR model.
        
        Args:
            batch: Batch data dictionary
            
        Returns:
            Adversarial audio tensor
        """
        print("Starting Stage 1 Attack...")
        
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
        
        for iteration in range(self.num_iter_stage1):
            optimizer.zero_grad()
            
            # Apply perturbations with bounds and rescaling
            bounded_delta = torch.clamp(delta, -self.initial_bound, self.initial_bound)
            scaled_delta = bounded_delta * rescale
            perturbed_audio = (scaled_delta * masks + audios).clamp(-32768, 32767)
            
            # Compute loss for each example in batch
            total_loss = 0
            
            for i in range(min(self.batch_size, audios.shape[0])):
                try:
                    # Loss must be computed on the tensor with gradients
                    loss = self.asr_model.compute_loss(perturbed_audio[i, :lengths[i]], target_texts[i])
                    total_loss += loss
                    
                    # Check if attack succeeded
                    if iteration % 10 == 0:
                        # Detach for transcription and other non-gradient operations
                        audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                        pred = self.asr_model.transcribe(audio_sample_np)
                        if pred.lower() == target_texts[i].lower():
                            print(f"SUCCESS at iteration {iteration} for example {i}")
                            # Update rescale factor
                            current_max = torch.max(torch.abs(bounded_delta[i]))
                            if rescale[i] * self.initial_bound > current_max:
                                rescale[i] = current_max / self.initial_bound
                            rescale[i] *= 0.8
                            
                            # Save best example
                            best_deltas[i] = perturbed_audio[i].clone()
                            
                except Exception as e:
                    print(f"Error processing example {i}: {e}")
                    continue
            
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Stage 1 - Iteration {iteration}, Loss: {total_loss:.4f}")
        
        # Use best deltas or final perturbed audio
        final_audio = torch.zeros_like(audios)
        for i in range(min(self.batch_size, audios.shape[0])):
            if best_deltas[i] is not None:
                final_audio[i] = best_deltas[i]
            else:
                # Fallback to final iteration
                bounded_delta = torch.clamp(delta, -self.initial_bound, self.initial_bound)
                scaled_delta = bounded_delta * rescale
                final_audio[i] = (scaled_delta[i] * masks[i] + audios[i]).clamp(-32768, 32767)
        
        print("Stage 1 completed.")
        return final_audio
    
    def run_attack(self, data_file: str, root_dir: str = "./", output_dir: str = "./output"):
        """
        Run the complete adversarial attack.
        
        Args:
            data_file: Path to data file containing audio files and targets
            root_dir: Root directory for input audio files
            output_dir: Directory to save adversarial examples
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
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
                stage1_results = self.stage1_attack(batch)
                
                # Save stage 1 results
                for i, audio_file in enumerate(audio_files):
                    if i < stage1_results.shape[0]:
                        name = Path(audio_file).stem
                        output_file = output_path / f"{name}_stage1.wav"
                        audio_data = stage1_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                        save_audio_file(audio_data, str(output_file), 16000)
                        
                        # Calculate distortion
                        original_audio = batch['audios'][i, :batch['lengths'][i]].cpu().numpy()
                        adversarial_audio = stage1_results[i, :batch['lengths'][i]].detach().cpu().numpy()
                        distortion = np.max(np.abs(adversarial_audio - original_audio))
                        print(f"Stage 1 distortion for {name}: {distortion:.2f}")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx // self.batch_size + 1}: {e}")
                continue


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
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for computation')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Initialize attack
    attack = AdversarialAttack(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run attack
    attack.run_attack(args.data_file, args.root_dir, args.output_dir)


if __name__ == "__main__":
    main()
