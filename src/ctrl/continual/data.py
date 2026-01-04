from __future__ import annotations
import numpy as np
import torch

def generate_sine(amplitude: float, phase: float, n: int = 100, device: torch.device | None = None):
    x = np.random.uniform(-5, 5, (n, 1)).astype(np.float32)
    y = (amplitude * np.sin(x + phase)).astype(np.float32)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    if device is not None:
        xt = xt.to(device)
        yt = yt.to(device)
    return xt, yt
