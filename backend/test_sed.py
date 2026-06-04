import numpy as np
from panns_inference import SoundEventDetection
sed = SoundEventDetection(checkpoint_path=None, device='cpu')
audio = np.zeros((1, 32000))
out = sed.inference(audio)
print("Type of out:", type(out))
if isinstance(out, tuple):
    print("Length of out:", len(out))
    for i, x in enumerate(out):
        print(f"Shape of {i}:", x.shape)
else:
    print("Shape of out:", out.shape)
