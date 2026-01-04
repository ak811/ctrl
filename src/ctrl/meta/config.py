from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MetaConfig:
    iterations: int = 200
    k_shots: int = 5
    inner_lr: float = 0.05
    meta_lr: float = 0.02
    gamma: float = 0.99
    entropy_bonus: float = 0.01
    max_steps_per_episode: int = 500
    seed: int = 42
    frame_stack: int = 6
