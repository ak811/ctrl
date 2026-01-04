from __future__ import annotations
import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility (best-effort; RL is still messy)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
