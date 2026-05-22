import torch
import timm
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])
# It also might need others. Let's just bypass it for now.
# Wait, PyTorch allows bypassing via torch.serialization.add_safe_globals.
# Alternatively, we can use torch.load context or monkeypatch torch.load!
import builtins
import torch

original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from hsemotion.facial_emotions import HSEmotionRecognizer
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
print("Success!")
