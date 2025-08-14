"""
Master test suite for verifying gradient flow in adversarial audio attacks.
Tests both Whisper and Wav2Vec2 models to ensure proper gradient computation.
"""

import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial_asr_modern.audio_utils import test_whisper_gradients
from adversarial_asr_modern.ctc_audio_utils import test_wav2vec_gradients


def test_all_gradients():
    """Complete gradient test for both models."""
    print("="*60)
    print("GRADIENT FLOW TEST SUITE")
    print("="*60)
    
    # Test 1: Basic PyTorch gradients
    print("\n1. Testing basic PyTorch gradients...")
    x = torch.randn(10, requires_grad=True)
    y = x ** 2
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Basic PyTorch gradients broken!"
    print("✅ Basic PyTorch working")
    
    # Test 2: Whisper gradients
    print("\n2. Testing Whisper model...")
    try:
        whisper_ok = test_whisper_gradients()
    except Exception as e:
        print(f"❌ Whisper test failed with error: {e}")
        whisper_ok = False
    
    # Test 3: Wav2Vec2 gradients
    print("\n3. Testing Wav2Vec2 model...")
    try:
        wav2vec_ok = test_wav2vec_gradients()
    except Exception as e:
        print(f"❌ Wav2Vec2 test failed with error: {e}")
        wav2vec_ok = False
    
    # Test 4: Attack pipeline
    print("\n4. Testing attack gradient flow...")
    audio = torch.randn(16000, requires_grad=True)
    delta = torch.zeros_like(audio, requires_grad=True)
    perturbed = audio + delta
    
    # Should maintain gradients through addition
    dummy_loss = perturbed.sum()
    dummy_loss.backward()
    assert delta.grad is not None, "Attack perturbation gradients broken!"
    print("✅ Attack pipeline gradients working")
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Whisper: {'✅ PASS' if whisper_ok else '❌ FAIL'}")
    print(f"  Wav2Vec2: {'✅ PASS' if wav2vec_ok else '❌ FAIL'}")
    print("="*60)
    
    return whisper_ok and wav2vec_ok


def test_whisper_only():
    """Test only Whisper model gradients."""
    print("Testing Whisper model gradients...")
    try:
        return test_whisper_gradients()
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wav2vec_only():
    """Test only Wav2Vec2 model gradients."""
    print("Testing Wav2Vec2 model gradients...")
    try:
        return test_wav2vec_gradients()
    except Exception as e:
        print(f"❌ Wav2Vec2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_gradient_check():
    """Quick check to verify basic gradient computation."""
    print("Quick gradient check...")
    
    # Test PyTorch basics
    x = torch.randn(100, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    
    if x.grad is not None and x.grad.norm() > 0:
        print("✅ PyTorch gradients working")
        return True
    else:
        print("❌ PyTorch gradients not working")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test gradient flow for adversarial attacks")
    parser.add_argument("--whisper", action="store_true", help="Test only Whisper model")
    parser.add_argument("--wav2vec", action="store_true", help="Test only Wav2Vec2 model")
    parser.add_argument("--quick", action="store_true", help="Quick gradient check only")
    args = parser.parse_args()
    
    if args.quick:
        success = quick_gradient_check()
    elif args.whisper:
        success = test_whisper_only()
    elif args.wav2vec:
        success = test_wav2vec_only()
    else:
        success = test_all_gradients()
    
    if success:
        print("\n🎉 All gradient tests passed! Ready for adversarial attacks.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Fix the issues above before running attacks.")
        sys.exit(1)