#!/usr/bin/env python3
"""
Simple installation test that doesn't require LibriSpeech data.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"✅ Librosa {librosa.__version__}")
    except ImportError as e:
        print(f"❌ Librosa import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy import failed: {e}")
        return False
        
    return True

def test_ctc_model():
    """Test CTC model loading."""
    print("\nTesting CTC model loading...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    try:
        from adversarial_asr_modern.ctc_audio_utils import CTCASRModel
        
        print("Loading CTC model (this may take a minute)...")
        model = CTCASRModel('base', 'cpu')
        
        print("✅ CTC model loaded successfully!")
        info = model.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Parameters: {info['parameters']:,}")
        print(f"Device: {info['device']}")
        
        return True
        
    except Exception as e:
        print(f"❌ CTC model loading failed: {e}")
        return False

def test_gradient_flow():
    """Test gradient computation."""
    print("\nTesting gradient flow...")
    
    try:
        import torch
        import numpy as np
        
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from adversarial_asr_modern.ctc_audio_utils import CTCASRModel
        
        model = CTCASRModel('base', 'cpu')
        
        # Create dummy audio
        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
        audio_tensor = torch.from_numpy(dummy_audio).requires_grad_(True)
        
        # Test loss computation
        loss = model.compute_attack_loss(audio_tensor, "hello world")
        print(f"Loss computed: {loss.item():.4f}")
        
        # Test gradient computation
        loss.backward()
        grad_norm = audio_tensor.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 0:
            print("✅ Gradient flow working correctly!")
            return True
        else:
            print("⚠️  No gradients detected")
            return False
            
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("ADVERSARIAL ASR MODERN - INSTALLATION TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Package imports failed. Please install missing dependencies:")
        print("pip install torch torchaudio transformers datasets numpy scipy librosa soundfile openai-whisper matplotlib absl-py")
        return
    
    # Test model loading
    model_ok = test_ctc_model()
    
    if not model_ok:
        print("\n❌ Model loading failed. Check your internet connection and try again.")
        return
    
    # Test gradient flow
    gradient_ok = test_gradient_flow()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Package imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Model loading:   {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Gradient flow:   {'✅ PASS' if gradient_ok else '❌ FAIL'}")
    
    if imports_ok and model_ok and gradient_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python run_ctc_attack.py --test_model")
        print("  python run_ctc_attack.py --num_examples 3")
    else:
        print("\n⚠️  Some tests failed. See SETUP_GUIDE.md for troubleshooting.")

if __name__ == "__main__":
    main()
