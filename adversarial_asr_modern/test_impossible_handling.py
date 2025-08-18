#!/usr/bin/env python3
"""
Test that impossible examples (inf/nan loss) are handled correctly.
"""

import numpy as np
import torch
from src.adversarial_asr_modern.adversarial_attack import AdversarialAttack
from src.adversarial_asr_modern.audio_utils import WhisperASRModel

def create_impossible_target():
    """Create a target that will likely cause tokenization issues."""
    # Mix of special characters, emojis, and very long repetitions
    return "🎵" * 100 + "∞" * 100 + "→" * 100

def test_impossible_handling():
    print("Testing handling of impossible examples...")
    
    # Create synthetic audio
    sample_rate = 16000
    duration = 2
    batch_size = 4
    
    # Generate synthetic audio
    audio_batch = [np.random.randn(sample_rate * duration) * 0.01 for _ in range(batch_size)]
    
    # Create a custom WhisperASRModel that returns inf for certain targets
    class TestWhisperModel(WhisperASRModel):
        def compute_loss(self, audio_tensor, target_text, sample_rate=16000):
            # Return inf for targets containing special patterns
            if "FORCE_INF" in target_text or len(target_text) > 500:
                print(f"  [TEST] Forcing inf loss for target: '{target_text[:30]}...'")
                return torch.tensor(float('inf'), device=self.device, requires_grad=True)
            return super().compute_loss(audio_tensor, target_text, sample_rate)
    
    # Create batch with guaranteed impossible targets
    batch = {
        'audios': torch.stack([torch.from_numpy(a).float() for a in audio_batch]),
        'original_texts': ['test one', 'test two', 'test three', 'test four'],
        'target_texts': [
            'hello world',  # Should work
            'FORCE_INF_TARGET',  # Will return inf
            'simple test',  # Should work
            'FORCE_INF_ANOTHER'  # Will return inf
        ],
        'masks': torch.ones(batch_size, sample_rate * duration),
        'lengths': [sample_rate * duration] * batch_size
    }
    
    # Create attack with custom model
    attack = AdversarialAttack(
        model_name='openai/whisper-base',
        batch_size=batch_size,
        num_iter_stage1=10,
        initial_bound=0.03,
        log_interval=5
    )
    
    # Replace model with test model
    attack.asr_model = TestWhisperModel()
    
    # Run attack
    print('\nRunning Stage 1 attack with forced impossible examples...')
    adv_audio, results = attack.stage1_attack(batch)
    
    # Check results
    print('\n' + '=' * 60)
    print('VALIDATION OF IMPOSSIBLE EXAMPLE HANDLING:')
    print('=' * 60)
    
    errors = []
    for r in results:
        idx = r['example_idx']
        target = r['target_text']
        success_iter = r['success_iteration']
        success_flag = r['success']
        
        print(f"\nExample {idx}:")
        print(f"  Target: '{target[:30]}...'")
        print(f"  Success iteration: {success_iter}")
        print(f"  Success flag: {success_flag}")
        
        # Validate based on target
        if 'FORCE_INF' in target:
            # Should be impossible
            if success_iter != -999:
                errors.append(f"Example {idx} should be impossible but has success_iteration={success_iter}")
            if success_flag:
                errors.append(f"Example {idx} should be impossible but has success=True")
            print(f"  ✓ Correctly marked as IMPOSSIBLE")
        else:
            # Should be attempted normally
            if success_iter == -999:
                errors.append(f"Example {idx} incorrectly marked as impossible")
            print(f"  ✓ Correctly attempted (success={success_flag})")
    
    # Summary
    print('\n' + '=' * 60)
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("✅ ALL TESTS PASSED!")
        print("  - Impossible examples have success_iteration=-999")
        print("  - Impossible examples have success=False")
        print("  - Normal examples are not marked as impossible")
    print('=' * 60)

if __name__ == "__main__":
    test_impossible_handling()