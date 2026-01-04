from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TransferConfig:
    env: str = "pong"               # pong | snake | puckworld
    timesteps: int = 1_000_000
    seed: int = 0
    learning_rate: float = 2.5e-4
    n_steps: int = 128
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
