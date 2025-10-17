import torch
import numpy as np
print("✅ Torch and NumPy working!")
print("Device:", "MPS" if torch.backends.mps.is_available() else "CPU")
