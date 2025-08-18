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
import random
from datetime import datetime

from .audio_utils import (
    WhisperASRModel, ParallelWhisperASRModel, DataParallelWhisperModel,
    load_audio_file, save_audio_file, 
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
                 initial_bound: float = 0.03,
                 lr_stage1: float = 0.05,
                 lr_stage2: float = 0.005,
                 num_iter_stage1: int = 1000,
                 num_iter_stage2: int = 4000,
                 log_interval: int = 10,
                 verbose: bool = False,
                 save_audio: bool = False,
                 skip_stage2_on_failure: bool = True,
                 use_parallel: bool = True,
                 num_parallel_models: int = 2):
        
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
        self.skip_stage2_on_failure = skip_stage2_on_failure
        self.use_parallel = use_parallel
        self.num_parallel_models = num_parallel_models

        # For reproducibility
        random.seed(17)  
        np.random.seed(17)  
        torch.manual_seed(17)  
        # Initialize ASR model (DataParallel, parallel threading, or single)
        if use_parallel and batch_size > 1:
            # Try DataParallel first (best for GPU utilization)
            if str(self.device).startswith('cuda'):
                print(f"Using DataParallel for better GPU utilization")
                self.asr_model = DataParallelWhisperModel(
                    model_name=model_name, 
                    device=self.device
                )
                self.parallel_mode = True
                self.parallel_type = 'dataparallel'
            else:
                # Fall back to threading-based parallel for non-CUDA
                print(f"Using thread-based parallel processing with {num_parallel_models} Whisper models")
                self.asr_model = ParallelWhisperASRModel(
                    num_models=num_parallel_models, 
                    model_name=model_name, 
                    device=self.device
                )
                self.parallel_mode = True
                self.parallel_type = 'threading'
        else:
            if use_parallel and batch_size <= 1:
                print("Parallel mode disabled: batch_size <= 1")
            self.asr_model = WhisperASRModel(model_name, device=self.device)
            self.parallel_mode = False
            self.parallel_type = None
        
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
        print(f"Batch size: {self.batch_size}")
        
        # WARNING for large batch sizes
        if self.batch_size > 20:
            print("\n" + "⚠️ " * 20)
            print(f"WARNING: Batch size {self.batch_size} is VERY LARGE!")
            print(f"This will require {self.batch_size} SEQUENTIAL Whisper forward passes per iteration.")
            print(f"Estimated time per iteration: {self.batch_size * 0.1:.1f}-{self.batch_size * 0.2:.1f} seconds")
            print(f"Total estimated time for {self.num_iter_stage1} iterations: {self.batch_size * 0.15 * self.num_iter_stage1 / 3600:.1f} hours")
            print(f"STRONGLY RECOMMEND: Reduce batch_size to 5-10 for reasonable performance.")
            print("⚠️ " * 20 + "\n")
        
        print(f"Logging predictions every {self.log_interval} iterations")
        print(f"Verbose mode: {self.verbose}")
        print("=" * 60)
        
        # Track timing
        import time
        start_time = time.time()
        iteration_times = []
        
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
        
        # Track progress over iterations
        iteration_history = {
            'iterations': [],
            'losses': [],
            'transcriptions': [],
            'perturbation_stats': []
        }
        
        # Initial transcriptions
        print("\n[INITIAL STATE]")
        for i, (original_text, target_text) in enumerate(zip(original_texts, target_texts)):
            if i < audios.shape[0]:
                audio_np = audios[i, :lengths[i]].cpu().numpy()
                try:
                    pred = self.asr_model.transcribe(audio_np)
                    if not pred or not pred.strip():
                        pred = "[empty]"
                except Exception as e:
                    pred = f"[error: {str(e)[:30]}]"
                print(f"Example {i}:")
                print(f"  Original: '{original_text}'")
                print(f"  Target:   '{target_text}'")
                print(f"  Current:  '{pred}'")
                print()
        
        # Main optimization loop
        for iteration in range(self.num_iter_stage1):
            iter_start = time.time()
            timing_breakdown = {}  # Track where time is spent
            optimizer.zero_grad()
            
            # Apply perturbations with bounds and rescaling
            bounded_delta = torch.clamp(delta, -self.initial_bound, self.initial_bound)
            scaled_delta = bounded_delta * rescale
            perturbed_audio = (scaled_delta * masks + audios).clamp(-1.0, 1.0)
            
            # Compute loss for all examples
            loss_start = time.time()
            try:
                batch_size_actual = min(self.batch_size, audios.shape[0])
                
                # Skip batched computation since it's not faster for Whisper
                # Try batched computation first (fastest)
                if False and hasattr(self.asr_model, 'compute_loss_batch') and batch_size_actual > 1:
                    # BATCHED PROCESSING - Single forward pass for all examples!
                    if iteration == 0:
                        print(f"⚡ Using BATCHED processing for batch size {batch_size_actual}")
                        print(f"   Computing all {batch_size_actual} losses in a single forward pass")
                    
                    batch_start = time.time()
                    # Prepare audio batch
                    audio_list = [perturbed_audio[i, :lengths[i]] for i in range(batch_size_actual)]
                    losses = self.asr_model.compute_loss_batch(audio_list, target_texts[:batch_size_actual])
                    batch_time = time.time() - batch_start
                    
                    if iteration % 100 == 0:
                        print(f"   [DEBUG] Batched loss computation took {batch_time:.3f}s for {batch_size_actual} examples")
                
                # Use parallel processing if available
                elif self.parallel_mode and batch_size_actual > 1:
                    if iteration == 0:
                        if self.parallel_type == 'dataparallel':
                            print(f"⚡ Using DataParallel for batch size {batch_size_actual}")
                            print(f"   GPU-optimized parallel processing")
                        else:
                            print(f"🚀 Using thread-based parallel with {self.num_parallel_models} models for batch size {batch_size_actual}")
                    
                    # Compute all losses in parallel
                    parallel_start = time.time()
                    
                    if self.parallel_type == 'dataparallel':
                        # Use DataParallel's optimized method
                        audio_list = [perturbed_audio[i, :lengths[i]] for i in range(batch_size_actual)]
                        losses = self.asr_model.compute_loss_parallel(audio_list, target_texts[:batch_size_actual])
                    else:
                        # Use threading-based parallel
                        losses = self.asr_model.compute_losses_parallel(
                            perturbed_audio[:batch_size_actual],
                            target_texts[:batch_size_actual],
                            lengths[:batch_size_actual]
                        )
                    
                    parallel_time = time.time() - parallel_start
                    if iteration % 100 == 0:
                        method = "DataParallel" if self.parallel_type == 'dataparallel' else "Threading"
                        print(f"   [DEBUG] {method} loss computation took {parallel_time:.3f}s for {batch_size_actual} examples")
                else:
                    # SEQUENTIAL PROCESSING - Original behavior
                    losses = []
                    
                    # WARNING: Large batch sizes will be very slow with sequential processing!
                    if iteration == 0:
                        print(f"📊 Using SEQUENTIAL processing for batch size {batch_size_actual}")
                        print(f"   Parallel mode: {self.parallel_mode}, Batch > 1: {batch_size_actual > 1}")
                        if batch_size_actual > 20 and not self.parallel_mode:
                            print(f"⚠️  WARNING: Batch size {batch_size_actual} with sequential loss computation will be VERY SLOW!")
                            print(f"   Each iteration will require {batch_size_actual} sequential Whisper forward passes.")
                            print(f"   Consider enabling parallel mode or reducing batch_size to 5-10.")
                    
                    # Process in smaller sub-batches if needed for memory
                    sub_batch_size = min(50, batch_size_actual)
                    
                    for sub_batch_start in range(0, batch_size_actual, sub_batch_size):
                        sub_batch_end = min(sub_batch_start + sub_batch_size, batch_size_actual)
                        sub_batch_audio = perturbed_audio[sub_batch_start:sub_batch_end]
                        sub_batch_lengths = lengths[sub_batch_start:sub_batch_end]
                        sub_batch_texts = target_texts[sub_batch_start:sub_batch_end]
                        
                        # Compute losses for this sub-batch (SEQUENTIAL - BOTTLENECK!)
                        for i, (audio, length, text) in enumerate(zip(sub_batch_audio, sub_batch_lengths, sub_batch_texts)):
                            idx = sub_batch_start + i
                            try:
                                loss = self.asr_model.compute_loss(audio[:length], text)
                                losses.append(loss)
                            except Exception as e:
                                print(f"Error processing example {idx}: {e}")
                                # Create a dummy loss that doesn't affect gradients
                                losses.append(torch.tensor(float('inf'), device=self.device, requires_grad=False))
                
                # Sum all losses
                valid_losses = [l for l in losses if not torch.isinf(l)]
                if valid_losses:
                    total_loss = torch.stack(valid_losses).sum()
                else:
                    total_loss = torch.tensor(0.0, device=self.device)
                
                individual_losses = [l.item() if not torch.isinf(l) else float('inf') for l in losses]
                
            except Exception as e:
                print(f"Error in batch loss computation: {e}")
                total_loss = torch.tensor(0.0, device=self.device)
                individual_losses = [float('inf')] * min(self.batch_size, audios.shape[0])
            
            timing_breakdown['loss_computation'] = time.time() - loss_start
            
            if total_loss > 0:
                total_loss.backward()
                
                if delta.grad is not None:
                    # Option 1: Sign the gradients (original method)
                    delta.grad.sign_()
                    
                optimizer.step()
            
            # Simplified logging intervals to avoid redundancy
            # Light progress every 10 iterations (just loss, no transcription)
            should_show_progress = (iteration % self.log_interval == 0) or (iteration == self.num_iter_stage1 - 1)
            # Full logging with transcriptions and history every 100 iterations
            should_full_log = (iteration % 100 == 0) or (iteration == self.num_iter_stage1 - 1)
            
            # Early stopping check - if all examples succeeded, we can stop
            if all(s != -1 for s in success_iterations[:min(self.batch_size, audios.shape[0])]):
                print(f"\n[Iteration {iteration}] All examples succeeded! Stopping early.")
                break
            
            # Track iteration time
            iteration_times.append(time.time() - iter_start)
            
            # Full logging with history and transcriptions every 100 iterations
            if should_full_log:
                history_entry = {
                    'iteration': iteration,
                    'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
                    'individual_losses': individual_losses.copy(),
                    'transcriptions': [],
                    'perturbation_stats': []
                }
                
                # Get transcriptions and stats (limit to 10 for large batches)
                max_history_examples = 10 if batch_size_actual > 20 else batch_size_actual
                for i in range(min(max_history_examples, audios.shape[0])):
                    audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                    
                    # Try to get transcription, handle both exceptions and empty results
                    try:
                        pred = self.asr_model.transcribe(audio_sample_np)
                        # Check if transcription is empty or just whitespace
                        if not pred or not pred.strip():
                            pred = "[EMPTY_TRANSCRIPTION]"
                    except Exception as e:
                        pred = f"[ERROR: {str(e)[:50]}]"
                        if self.verbose:
                            print(f"Warning: Transcription failed for example {i}: {e}")
                    
                    perturbation = (perturbed_audio[i, :lengths[i]] - audios[i, :lengths[i]]).detach()
                    max_pert = torch.max(torch.abs(perturbation)).item()
                    mean_pert = torch.mean(torch.abs(perturbation)).item()
                    
                    # Only mark as success if we got a real transcription that matches
                    success = (pred.lower().strip() == target_texts[i].lower().strip() 
                              and not pred.startswith("["))
                    
                    history_entry['transcriptions'].append({
                        'example_idx': i,
                        'original': original_texts[i],
                        'target': target_texts[i],
                        'prediction': pred,
                        'success': success
                    })
                    
                    history_entry['perturbation_stats'].append({
                        'example_idx': i,
                        'max_perturbation': max_pert,
                        'mean_perturbation': mean_pert
                    })
                
                iteration_history['iterations'].append(iteration)
                iteration_history['losses'].append(history_entry['individual_losses'])
                iteration_history['transcriptions'].append(history_entry['transcriptions'])
                iteration_history['perturbation_stats'].append(history_entry['perturbation_stats'])
                
                if self.verbose:
                    print(f"[Iteration {iteration}] Saved history checkpoint")
            
            # Light progress indicator (every 10 iterations - no transcription)
            if should_show_progress and not should_full_log:
                avg_time = np.mean(iteration_times[-min(50, len(iteration_times)):])
                est_remaining = avg_time * (self.num_iter_stage1 - iteration - 1)
                loss_time = timing_breakdown.get('loss_computation', 0)
                print(f"[Iteration {iteration}/{self.num_iter_stage1}] Loss: {total_loss:.4f} [{avg_time:.2f}s/iter, Loss comp: {loss_time:.4f}s, ETA: {est_remaining/60:.1f}min]")
            
            # Full logging with transcriptions (every 100 iterations)
            if should_full_log:
                avg_time = np.mean(iteration_times[-min(50, len(iteration_times)):])
                est_remaining = avg_time * (self.num_iter_stage1 - iteration - 1)
                print(f"\n[Iteration {iteration}/{self.num_iter_stage1}] [{avg_time:.2f}s/iter, ETA: {est_remaining/60:.1f}min]")
                print(f"Total Loss: {total_loss:.4f}")
                
                # Check predictions for all examples with transcription
                for i in range(min(self.batch_size, audios.shape[0])):
                    if success_iterations[i] != -1:
                        # Already succeeded, skip detailed logging
                        print(f"Example {i}: SUCCESS (achieved at iteration {success_iterations[i]})")
                        continue
                    
                    try:
                        # Always transcribe during full logging
                        audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                        pred = self.asr_model.transcribe(audio_sample_np)
                        
                        # Handle empty transcriptions
                        if not pred or not pred.strip():
                            pred = "[empty]"
                            success = False
                        else:
                            # Check for success
                            success = pred.lower().strip() == target_texts[i].lower().strip()
                        
                        # Calculate perturbation stats
                        perturbation = (perturbed_audio[i, :lengths[i]] - audios[i, :lengths[i]]).detach()
                        max_pert = torch.max(torch.abs(perturbation)).item()
                        mean_pert = torch.mean(torch.abs(perturbation)).item()
                        
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
            
            # Check for early success on low-loss examples (DISABLED for large batches)
            elif batch_size_actual <= 20 and any(loss < 2.0 for loss in individual_losses):
                # Only do opportunistic checking for small batches
                for i in range(min(self.batch_size, audios.shape[0])):
                    if success_iterations[i] == -1 and individual_losses[i] < 2.0:
                        try:
                            # Quick check for potential success
                            audio_sample_np = perturbed_audio[i, :lengths[i]].detach().cpu().numpy()
                            pred = self.asr_model.transcribe(audio_sample_np)
                            if pred and pred.lower().strip() == target_texts[i].lower().strip():
                                success_iterations[i] = iteration
                                # Calculate perturbation stats
                                perturbation = (perturbed_audio[i, :lengths[i]] - audios[i, :lengths[i]]).detach()
                                max_pert = torch.max(torch.abs(perturbation)).item()
                                
                                print(f"\n[Iteration {iteration}] Example {i}: ✓ SUCCESS!")
                                print(f"  Target achieved: '{target_texts[i]}'")
                                print(f"  Loss: {individual_losses[i]:.4f}, Max pert: {max_pert:.2f}")
                                
                                # Update rescale factor
                                current_max = torch.max(torch.abs(bounded_delta[i]))
                                if rescale[i] * self.initial_bound > current_max:
                                    rescale[i] = current_max / self.initial_bound
                                rescale[i] *= 0.8
                                
                                # Save best example
                                best_deltas[i] = perturbed_audio[i].clone()
                        except:
                            pass  # Silent fail for opportunistic checks
        
        # Final summary and collect results
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("STAGE 1 COMPLETE - FINAL RESULTS:")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"Average time per iteration: {np.mean(iteration_times):.2f} seconds")
        
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
            try:
                final_pred = self.asr_model.transcribe(audio_sample_np)
                # Handle empty transcriptions
                if not final_pred or not final_pred.strip():
                    final_pred = "[EMPTY_TRANSCRIPTION]"
            except Exception as e:
                final_pred = f"[ERROR: {str(e)[:50]}]"
                print(f"Warning: Final transcription failed for example {i}: {e}")
            
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
                'stage': 'stage1',
                'iteration_history': iteration_history if i == 0 else None  # Add history only to first result to avoid duplication
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
    
    def stage2_attack(self, batch: Dict, stage1_audio: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Stage 2: Refine perturbations to be imperceptible using psychoacoustic masking.
        
        This stage jointly optimizes:
        1. Cross-entropy loss to maintain the adversarial transcription
        2. Masking threshold loss to ensure imperceptibility
        
        Args:
            batch: Batch data dictionary
            stage1_audio: Adversarial audio from Stage 1
            
        Returns:
            Tuple of (adversarial audio tensor, attack results)
        """
        print("=" * 60)
        print("Starting Stage 2 Attack (Imperceptibility Refinement)...")
        print(f"Logging predictions every {self.log_interval} iterations")
        print("=" * 60)
        
        audios = batch['audios']
        target_texts = batch['target_texts']
        th_batch = batch['th_batch']
        psd_max_batch = batch['psd_max_batch']
        masks = batch['masks']
        lengths = batch['lengths']
        
        # Initialize delta from Stage 1 results
        stage1_delta = stage1_audio - audios
        delta = stage1_delta.clone().detach().requires_grad_(True)
        
        # Initialize alpha parameters for balancing losses
        alpha = torch.ones(self.batch_size, device=self.device) * 0.05
        min_alpha = 0.0005
        
        # Optimizer with lower learning rate for Stage 2
        optimizer = optim.Adam([delta], lr=self.lr_stage2)
        
        # Track best results
        best_loss_th = [float('inf')] * self.batch_size
        best_deltas = [None] * self.batch_size
        best_alphas = [None] * self.batch_size
        attack_results = []
        
        # Initial check
        print("\n[STAGE 2 INITIAL STATE]")
        for i in range(min(self.batch_size, audios.shape[0])):
            perturbed = (audios[i] + delta[i] * masks[i]).clamp(-1.0, 1.0)
            audio_np = perturbed[:lengths[i]].detach().cpu().numpy()
            try:
                pred = self.asr_model.transcribe(audio_np)
                if not pred or not pred.strip():
                    pred = "[empty]"
            except Exception as e:
                pred = f"[error: {str(e)[:30]}]"
            print(f"Example {i}:")
            print(f"  Target:  '{target_texts[i]}'")
            print(f"  Current: '{pred}'")
            
        # Main optimization loop
        for iteration in range(self.num_iter_stage2):
            optimizer.zero_grad()
            
            # Apply perturbations with masking
            perturbed_audio = (delta * masks + audios).clamp(-1.0, 1.0)
            
            # Compute ASR loss
            total_asr_loss = 0
            individual_asr_losses = []
            
            for i in range(min(self.batch_size, audios.shape[0])):
                try:
                    loss = self.asr_model.compute_loss(perturbed_audio[i, :lengths[i]], target_texts[i])
                    total_asr_loss += loss
                    individual_asr_losses.append(loss.item())
                except Exception as e:
                    print(f"Error computing ASR loss for example {i}: {e}")
                    individual_asr_losses.append(float('inf'))
            
            # Compute masking threshold loss
            total_th_loss = 0
            individual_th_losses = []
            
            for i in range(min(self.batch_size, audios.shape[0])):
                # Compute PSD of the perturbation
                perturbation = delta[i, :lengths[i]] * masks[i, :lengths[i]]
                
                # Use Transform to compute PSD
                # Convert psd_max to tensor if needed
                psd_max_tensor = torch.tensor(psd_max_batch[i], dtype=torch.float32, device=self.device)
                psd_delta = self.transform(perturbation.unsqueeze(0), psd_max_tensor)
                
                # Convert threshold to tensor
                th_tensor = torch.from_numpy(th_batch[i]).float().to(self.device)
                
                # Compute threshold loss (penalize when exceeding threshold)
                # Match dimensions - psd_delta shape: [1, freq_bins], th_tensor shape: [time_frames, freq_bins]
                # Average over time dimension of threshold
                th_mean = th_tensor.mean(dim=0, keepdim=True)
                
                # Ensure matching dimensions
                if psd_delta.shape[-1] != th_mean.shape[-1]:
                    min_dim = min(psd_delta.shape[-1], th_mean.shape[-1])
                    psd_delta = psd_delta[:, :min_dim]
                    th_mean = th_mean[:, :min_dim]
                
                loss_th = torch.mean(torch.relu(psd_delta - th_mean))
                th_loss_weighted = alpha[i] * loss_th
                total_th_loss += th_loss_weighted
                individual_th_losses.append(loss_th.item())
            
            # Total loss combines ASR loss and threshold loss
            total_loss = total_asr_loss + total_th_loss
            
            if total_loss > 0:
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                optimizer.step()
            
            # Adaptive learning rate adjustment at iteration 3000
            if iteration == 3000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr_stage2 * 0.1
                print(f"\n[Learning rate reduced to {param_group['lr']:.6f}]")
            
            # Logging and alpha adjustment
            should_log = (iteration % self.log_interval == 0) or (iteration == self.num_iter_stage2 - 1)
            
            if should_log or (iteration % 20 == 0):
                if should_log:
                    print(f"\n[Iteration {iteration}/{self.num_iter_stage2}]")
                    print(f"Total ASR Loss: {total_asr_loss:.4f}, Total Th Loss: {total_th_loss:.4f}")
                
                for i in range(min(self.batch_size, audios.shape[0])):
                    # Get current prediction
                    perturbed = (audios[i] + delta[i] * masks[i]).clamp(-1.0, 1.0)
                    audio_sample_np = perturbed[:lengths[i]].detach().cpu().numpy()
                    try:
                        pred = self.asr_model.transcribe(audio_sample_np)
                        if not pred or not pred.strip():
                            pred = "[empty]"
                    except Exception as e:
                        pred = f"[error: {str(e)[:30]}]"
                    
                    # Check if maintaining target transcription
                    success = pred.lower().strip() == target_texts[i].lower().strip()
                    
                    if success:
                        # Save if this has lower threshold loss
                        if individual_th_losses[i] < best_loss_th[i]:
                            best_loss_th[i] = individual_th_losses[i]
                            best_deltas[i] = perturbed.clone()
                            best_alphas[i] = alpha[i].item()
                            
                            if should_log:
                                print(f"Example {i}: ✓ IMPROVED")
                                print(f"  Target: '{target_texts[i]}'")
                                print(f"  Maintained: '{pred}'")
                                print(f"  ASR Loss: {individual_asr_losses[i]:.4f}")
                                print(f"  Th Loss: {individual_th_losses[i]:.6f}")
                                print(f"  Alpha: {alpha[i].item():.6f}")
                        
                        # Increase alpha every 20 iterations to enforce stronger masking
                        if iteration % 20 == 0:
                            alpha[i] *= 1.2
                    else:
                        # Reduce alpha if failing to maintain transcription
                        if iteration % 50 == 0:
                            alpha[i] *= 0.8
                            alpha[i] = max(alpha[i], min_alpha)
                        
                        if should_log:
                            print(f"Example {i}: Failed")
                            print(f"  Target: '{target_texts[i]}'")
                            print(f"  Current: '{pred}'")
                            print(f"  Alpha: {alpha[i].item():.6f}")
                
                if should_log:
                    successful = sum(1 for b in best_deltas if b is not None)
                    print(f"\nProgress: {successful}/{min(self.batch_size, audios.shape[0])} with improved imperceptibility")
                    print("-" * 40)
        
        # Final results
        print("\n" + "=" * 60)
        print("STAGE 2 COMPLETE - FINAL RESULTS:")
        
        final_audio = torch.zeros_like(audios)
        
        for i in range(min(self.batch_size, audios.shape[0])):
            # Use best delta if available, otherwise use Stage 1 result
            if best_deltas[i] is not None:
                final_audio[i] = best_deltas[i]
            else:
                final_audio[i] = stage1_audio[i]
            
            # Get final prediction
            audio_sample_np = final_audio[i, :lengths[i]].detach().cpu().numpy()
            try:
                final_pred = self.asr_model.transcribe(audio_sample_np)
                # Handle empty transcriptions
                if not final_pred or not final_pred.strip():
                    final_pred = "[EMPTY_TRANSCRIPTION]"
            except Exception as e:
                final_pred = f"[ERROR: {str(e)[:50]}]"
                print(f"Warning: Final transcription failed for example {i}: {e}")
            
            # Calculate final stats
            perturbation = (final_audio[i, :lengths[i]] - audios[i, :lengths[i]]).detach()
            max_pert = torch.max(torch.abs(perturbation)).item()
            mean_pert = torch.mean(torch.abs(perturbation)).item()
            
            # Create result entry
            result = {
                'example_idx': i,
                'target_text': target_texts[i],
                'final_text': final_pred,
                'success': final_pred.lower().strip() == target_texts[i].lower().strip(),
                'final_loss_th': best_loss_th[i] if best_loss_th[i] != float('inf') else -1,
                'final_alpha': best_alphas[i] if best_alphas[i] else alpha[i].item(),
                'max_perturbation': max_pert,
                'mean_perturbation': mean_pert,
                'stage': 'stage2'
            }
            attack_results.append(result)
            
            # Print summary
            if result['success']:
                print(f"Example {i}: SUCCESS")
                print(f"  Final Th Loss: {result['final_loss_th']:.6f}")
                print(f"  Final Alpha: {result['final_alpha']:.6f}")
            else:
                print(f"Example {i}: FAILED to maintain target")
                print(f"  Target: '{target_texts[i]}'")
                print(f"  Final: '{final_pred}'")
        
        print("=" * 60)
        
        return final_audio, attack_results
    
    def stage2_attack_selective(self, batch: Dict, stage1_audio: torch.Tensor, 
                                stage1_results: List[Dict], successful_indices: List[int]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Run Stage 2 only on successful Stage 1 examples.
        
        Args:
            batch: Original batch data
            stage1_audio: All audio from Stage 1
            stage1_results: Results from Stage 1
            successful_indices: Indices of successful examples to process
            
        Returns:
            Tuple of (audio tensor with Stage 2 for successful examples, Stage 2 results)
        """
        if not successful_indices:
            return stage1_audio, []
        
        print(f"\nStage 2: Processing {len(successful_indices)} successful examples")
        
        # Extract only successful examples for Stage 2
        audios = batch['audios']
        lengths = batch['lengths']
        
        # Create sub-batch for successful examples
        sub_batch_audios = torch.stack([audios[i] for i in successful_indices])
        sub_batch_targets = [batch['target_texts'][i] for i in successful_indices]
        sub_batch_th = [batch['th_batch'][i] for i in successful_indices]
        sub_batch_psd_max = np.array([batch['psd_max_batch'][i] for i in successful_indices])
        sub_batch_masks = torch.stack([batch['masks'][i] for i in successful_indices])
        sub_batch_lengths = [lengths[i] for i in successful_indices]
        sub_batch_stage1 = torch.stack([stage1_audio[i] for i in successful_indices])
        
        # Create filtered batch
        filtered_batch = {
            'audios': sub_batch_audios,
            'original_texts': [batch['original_texts'][i] for i in successful_indices],
            'target_texts': sub_batch_targets,
            'th_batch': sub_batch_th,
            'psd_max_batch': sub_batch_psd_max,
            'masks': sub_batch_masks,
            'lengths': sub_batch_lengths,
            'max_length': batch['max_length']
        }
        
        # Run Stage 2 on filtered batch
        stage2_audio_filtered, stage2_results_filtered = self.stage2_attack(filtered_batch, sub_batch_stage1)
        
        # Reconstruct full results maintaining original indexing
        final_audio = stage1_audio.clone()
        stage2_results = []
        
        for orig_idx in range(len(stage1_results)):
            if orig_idx in successful_indices:
                # Get the position in the filtered results
                filtered_idx = successful_indices.index(orig_idx)
                
                # Update audio for this example
                final_audio[orig_idx] = stage2_audio_filtered[filtered_idx]
                
                # Add Stage 2 result
                if filtered_idx < len(stage2_results_filtered):
                    result = stage2_results_filtered[filtered_idx].copy()
                    result['example_idx'] = orig_idx  # Maintain original index
                    stage2_results.append(result)
            else:
                # For failed Stage 1, keep Stage 1 audio and no Stage 2 result
                # Audio already in final_audio from clone
                pass
        
        print(f"Stage 2 selective processing complete: {len(stage2_results)} results")
        
        return final_audio, stage2_results
    
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
                stage1_audio, stage1_results = self.stage1_attack(batch)
                
                # Check Stage 1 success and conditionally run Stage 2
                stage1_successes = [r.get('success', False) for r in stage1_results]
                any_success = any(stage1_successes)
                
                if self.skip_stage2_on_failure and not any_success:
                    # Skip Stage 2 entirely if no Stage 1 succeeded
                    print("\n" + "=" * 60)
                    print("SKIPPING STAGE 2 - No successful Stage 1 attacks")
                    print("=" * 60)
                    stage2_audio = stage1_audio
                    stage2_results = []
                elif self.skip_stage2_on_failure and any_success:
                    # Run Stage 2 only on successful examples
                    successful_count = sum(stage1_successes)
                    print("\n" + "=" * 60)
                    print(f"Running Stage 2 on {successful_count}/{len(stage1_results)} successful examples")
                    print("=" * 60)
                    
                    # Create filtered batch for Stage 2
                    successful_indices = [i for i, success in enumerate(stage1_successes) if success]
                    stage2_audio, stage2_results = self.stage2_attack_selective(
                        batch, stage1_audio, stage1_results, successful_indices
                    )
                else:
                    # Original behavior - run Stage 2 on all examples
                    print("\nRunning Stage 2 on all examples (skip_stage2_on_failure=False)")
                    stage2_audio, stage2_results = self.stage2_attack(batch, stage1_audio)
                
                # Process and save results for both stages
                for i, audio_file in enumerate(audio_files):
                    if i < len(stage1_results):
                        name = Path(audio_file).stem
                        
                        # Add file information to Stage 1 results
                        stage1_results[i]['audio_file'] = audio_file
                        stage1_results[i]['audio_name'] = name
                        
                        # Calculate Stage 1 distortion
                        if i < stage1_audio.shape[0]:
                            original_audio = batch['audios'][i, :batch['lengths'][i]].cpu().numpy()
                            stage1_adversarial = stage1_audio[i, :batch['lengths'][i]].detach().cpu().numpy()
                            stage1_distortion = np.max(np.abs(stage1_adversarial - original_audio))
                            stage1_results[i]['stage1_distortion'] = stage1_distortion
                            print(f"Stage 1 distortion for {name}: {stage1_distortion:.4f}")
                            
                            # Save Stage 1 audio if requested
                            if self.save_audio:
                                output_file = output_path / f"{name}_stage1.wav"
                                save_audio_file(stage1_adversarial, str(output_file), 16000)
                        
                        # Add Stage 2 results if available
                        if i < len(stage2_results):
                            # Merge Stage 2 information into results
                            stage2_results[i]['audio_file'] = audio_file
                            stage2_results[i]['audio_name'] = name
                            stage2_results[i]['original_text'] = stage1_results[i].get('original_text', '')
                            
                            # Calculate Stage 2 distortion
                            if i < stage2_audio.shape[0]:
                                stage2_adversarial = stage2_audio[i, :batch['lengths'][i]].detach().cpu().numpy()
                                stage2_distortion = np.max(np.abs(stage2_adversarial - original_audio))
                                stage2_results[i]['stage2_distortion'] = stage2_distortion
                                print(f"Stage 2 distortion for {name}: {stage2_distortion:.4f}")
                                
                                # Save Stage 2 audio if requested
                                if self.save_audio:
                                    output_file = output_path / f"{name}_stage2.wav"
                                    save_audio_file(stage2_adversarial, str(output_file), 16000)
                
                # Collect results from both stages
                # Combine Stage 1 and Stage 2 results
                combined_results = []
                for i in range(len(audio_files)):
                    result = {}
                    
                    # Start with Stage 1 results
                    if i < len(stage1_results):
                        result.update(stage1_results[i])
                    
                    # Override/add Stage 2 results if available
                    if i < len(stage2_results):
                        # Keep Stage 1 info but update with Stage 2 final results
                        result['stage1_success'] = stage1_results[i].get('success', False)
                        result['stage1_distortion'] = stage1_results[i].get('stage1_distortion', -1)
                        result.update(stage2_results[i])
                        result['final_stage'] = 'stage2'
                    else:
                        result['final_stage'] = 'stage1'
                    
                    combined_results.append(result)
                
                all_results.extend(combined_results)
                
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
        
        # Extract iteration history from results (if present)
        # Remove from ALL results to avoid CSV corruption when processing multiple batches
        iteration_history = None
        for result in results:
            if result.get('iteration_history'):
                if iteration_history is None:  # Keep the first one for saving
                    iteration_history = result['iteration_history']
                # Remove from ALL results to avoid CSV corruption
                del result['iteration_history']
        # Note: No break statement - we process ALL results
        
        if filepath.suffix == '.json':
            # For JSON, we can include the iteration history
            if iteration_history:
                results[0]['iteration_history'] = iteration_history
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filepath} (JSON format)")
        else:
            # Default to CSV
            df = pd.DataFrame(results)
            df.to_csv(filepath, index=False)
            print(f"\nResults saved to {filepath} (CSV format)")
            
            # Save iteration history to separate file if present
            if iteration_history:
                history_filepath = filepath.with_suffix('').with_name(filepath.stem + '_history.json')
                with open(history_filepath, 'w') as f:
                    json.dump(iteration_history, f, indent=2)
                print(f"Iteration history saved to {history_filepath}")
    
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
