# Gradient Fix Checklist for Whisper & Wav2Vec2

## 🔧 Whisper Gradient Fixes

### 1. **Enable Gradient Computation**
```python
# In WhisperASRModel.__init__
self.model.train()  # ✅ Enable gradient computation
# Keep model weights frozen - we only want input gradients
for param in self.model.parameters():
    param.requires_grad = False
```

### 2. **Fix the Broken Gradient Chain**
```python
# ❌ WRONG - Current code breaks gradients:
audio_np = audio_cpu.detach().numpy()  # This breaks gradient flow!
inputs = self.processor(audio_np, ...)

# ✅ CORRECT - Keep everything in PyTorch:
# Option A: Use torchaudio for differentiable mel-spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
).to(self.device)
mel_spec = mel_transform(audio_tensor)  # Maintains gradients!

# Option B: Manual STFT (if torchaudio unavailable)
stft = torch.stft(audio_tensor, n_fft=400, hop_length=160, 
                  window=torch.hann_window(400).to(self.device),
                  return_complex=True)
```

### 3. **Proper Loss Computation**
```python
def compute_loss(self, audio_tensor: torch.Tensor, target_text: str):
    # Process features while maintaining gradients
    mel_spec = self.compute_mel_spectrogram_differentiable(audio_tensor)
    
    # Get target IDs (no gradients needed)
    with torch.no_grad():
        target_ids = self.processor.tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)
    
    # Forward pass WITH gradients
    outputs = self.model(input_features=mel_spec, labels=target_ids)
    return outputs.loss  # This will have gradients!
```

### 4. **Test Whisper Gradients**
```python
def test_whisper_gradients():
    print("Testing Whisper gradients...")
    model = WhisperASRModel(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test audio with gradients
    audio = torch.randn(16000, requires_grad=True, device=model.device)
    
    # Compute loss
    loss = model.compute_loss(audio, "test transcription")
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    
    if audio.grad is not None:
        grad_norm = audio.grad.norm().item()
        print(f"✅ Whisper gradients working! Norm: {grad_norm:.6f}")
        return True
    else:
        print("❌ No Whisper gradients detected!")
        return False
```

## 🔧 Wav2Vec2 Gradient Fixes

### 1. **Model Initialization**
```python
# In CTCASRModel.__init__
self.model.eval()  # Keep in eval mode but...
# Don't freeze parameters explicitly - wav2vec2 handles this
```

### 2. **Fix Input Processing**
```python
# ❌ WRONG - Raw audio directly to model:
outputs = self.model(audio_tensor)  # Won't work properly

# ✅ CORRECT - Proper input format:
def compute_loss_differentiable(self, audio_tensor: torch.Tensor, target_text: str):
    # Ensure proper shape [batch_size, sequence_length]
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Normalize to [-1, 1] (wav2vec2 requirement)
    if audio_tensor.abs().max() > 1.0:
        audio_tensor = audio_tensor / audio_tensor.abs().max()
    
    # Get target tokens
    with torch.no_grad():
        labels = self.processor.tokenizer(target_text, return_tensors="pt").input_ids
        labels = labels.to(self.device)
    
    # Forward pass - wav2vec2 handles feature extraction internally
    outputs = self.model(input_values=audio_tensor, labels=labels)
    return outputs.loss
```

### 3. **CTC Loss Computation**
```python
def compute_attack_loss(self, audio_tensor: torch.Tensor, target_text: str):
    # Get logits with gradients
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Forward pass
    outputs = self.model(input_values=audio_tensor)
    logits = outputs.logits
    
    # Manual CTC loss for better control
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Target preparation
    with torch.no_grad():
        targets = self.processor.tokenizer(target_text, return_tensors="pt").input_ids
        targets = targets.to(self.device)
    
    # Compute CTC loss
    input_lengths = torch.full((1,), logits.shape[1], device=self.device)
    target_lengths = torch.full((1,), targets.shape[1], device=self.device)
    
    loss = F.ctc_loss(
        log_probs.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=self.processor.tokenizer.pad_token_id,
        zero_infinity=True
    )
    
    return -loss  # Negative for attack
```

### 4. **Test Wav2Vec2 Gradients**
```python
def test_wav2vec_gradients():
    print("Testing Wav2Vec2 gradients...")
    model = CTCASRModel(model_size='base')
    
    # Create test audio
    audio = torch.randn(16000, requires_grad=True, device=model.device)
    
    # Compute loss
    loss = model.compute_attack_loss(audio, "hello world")
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    
    if audio.grad is not None:
        grad_norm = audio.grad.norm().item()
        print(f"✅ Wav2Vec2 gradients working! Norm: {grad_norm:.6f}")
        return True
    else:
        print("❌ No Wav2Vec2 gradients detected!")
        return False
```

## 🧪 Master Test Suite

### Run This Complete Test:
```python
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
    whisper_ok = test_whisper_gradients()
    
    # Test 3: Wav2Vec2 gradients
    print("\n3. Testing Wav2Vec2 model...")
    wav2vec_ok = test_wav2vec_gradients()
    
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

# Run the test
if __name__ == "__main__":
    success = test_all_gradients()
    if success:
        print("\n🎉 All gradient tests passed! Ready for adversarial attacks.")
    else:
        print("\n⚠️ Some tests failed. Fix the issues above before running attacks.")
```

## 📋 Common Pitfalls Checklist

- [ ] **Never use `.detach()` or `.numpy()` in the forward pass**
- [ ] **Always set `requires_grad=True` on input audio**
- [ ] **Use `model.train()` for Whisper (even if not training weights)**
- [ ] **Keep everything in PyTorch tensors throughout**
- [ ] **Normalize audio appropriately ([-1, 1] for Wav2Vec2)**
- [ ] **Check tensor shapes match model expectations**
- [ ] **Use `with torch.no_grad()` only for tokenization**
- [ ] **Verify loss is scalar and finite before `.backward()`**

## 🚀 Quick Debug Commands

```bash
# Test gradient flow quickly
python -c "from audio_utils import test_whisper_gradients; test_whisper_gradients()"
python -c "from ctc_audio_utils import test_wav2vec_gradients; test_wav2vec_gradients()"

# Check if CUDA is being used properly
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify model loads correctly
python -c "from transformers import WhisperForConditionalGeneration; m = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base'); print('Model loaded')"
```

## ✅ Confirmation Step

1. **Run the test suite** - All tests should pass
2. **Check gradient norms** - Should be > 0 but < 1000 (not exploding)
3. **Verify loss decreases** - Loss should decrease over iterations
4. **Monitor attack success** - Transcriptions should start changing after ~100 iterations
5. **Save checkpoint** - Save working code before making more changes!

Once all these checks pass, your adversarial attacks should work properly!