from __future__ import annotations
from copy import deepcopy
import torch

def reptile_update(meta_model: torch.nn.Module, adapted_model: torch.nn.Module, meta_lr: float):
    """Move meta params toward adapted params (Reptile)."""
    with torch.no_grad():
        for p_meta, p_adapt in zip(meta_model.parameters(), adapted_model.parameters()):
            p_meta.data.add_(meta_lr * (p_adapt.data - p_meta.data))

def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    m = deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    return m
