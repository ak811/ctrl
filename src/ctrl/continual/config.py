from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import List, Tuple

@dataclass(frozen=True)
class ContinualConfig:
    episodes: int = 20_000
    learning_rate: float = 1e-3
    log_interval: int = 2_000

    # MAML sine
    inner_steps: int = 5
    maml_inner_lr: float = 1e-2
    maml_meta_lr: float = 1e-3
    maml_num_tasks: int = 5
    eval_adapt_steps: int = 200

    # EWC
    ewc_lambda: float = 500.0

    # Experiment setup
    n_seeds: int = 3

    task_params: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 0.0), (2.0, math.pi/4), (0.5, math.pi/2),
        (1.5, math.pi/3), (0.8, math.pi/6),
        (3.5, 2.4), (0.2, 0.1), (4.8, 0.0),
    ])

    transfer_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=lambda: [
        ((1.0, 0.0), (2.0, math.pi/4)),
        ((2.0, math.pi/4), (0.5, math.pi/2)),
        ((0.5, math.pi/2), (3.5, 2.4)),
        ((1.5, math.pi/3), (0.2, 0.1)),
    ])
