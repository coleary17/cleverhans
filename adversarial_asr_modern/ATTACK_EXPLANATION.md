# Understanding Whisper Adversarial Attacks

## Why Teacher Forcing is Correct for Adversarial Attacks

Initially, I thought using teacher forcing (providing target labels) was wrong, but it's actually correct for adversarial attacks:

### How it Works:
1. **Forward Pass**: Audio → Mel-Spectrogram → Encoder → Decoder (with target labels) → Loss
2. **Backward Pass**: Loss gradients flow back through decoder → encoder → mel-spectrogram → audio
3. **Optimization**: We modify the AUDIO to minimize the loss of producing the TARGET text

### The Key Insight:
- We're NOT training the model (all model weights are frozen)
- We're training the INPUT AUDIO to produce a specific output
- The loss measures: "How hard is it for the model to produce the target text given this audio?"
- By minimizing this loss, we make the audio "sound like" the target text to the model

### Why Transcription Wasn't Changing:
The issue wasn't the loss function, but rather:

1. **Learning Rate Too Small**: With complex models like Whisper, gradients can be small
   - Need higher learning rates (0.01-0.1 instead of 0.001)
   
2. **Perturbation Bounds Too Small**: 
   - Whisper is robust, needs larger perturbations to affect output
   - Bounds of 0.05-0.1 work better than 0.01-0.02

3. **Optimization Strategy**:
   - Adam optimizer works well but needs proper learning rate
   - Gradient clipping helps stability
   - More iterations may be needed (2000-5000)

4. **Audio Normalization**:
   - Whisper expects audio in [-1, 1] range
   - Perturbations must respect this range

## The Correct Approach:

```python
# This IS correct for adversarial attacks:
outputs = model(input_features=mel_spec, labels=target_ids)
loss = outputs.loss  # This loss guides the audio to produce target text
```

The model uses the target labels to compute how well the current audio could produce that text. By minimizing this loss through gradient descent on the audio, we create adversarial examples.

## Common Misconceptions:

❌ "Teacher forcing is only for training models"
✅ Teacher forcing in adversarial attacks optimizes the INPUT, not the model

❌ "The loss should predict without labels"  
✅ The loss with labels measures compatibility between audio and target text

❌ "Small perturbations always work"
✅ Different models have different robustness; Whisper needs larger perturbations