from __future__ import annotations
from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import SineMLP
from .data import generate_sine
from ..common.io import write_csv, ensure_dir

mse = nn.MSELoss()

def train_maml_sine(
    seed: int,
    episodes: int,
    inner_steps: int,
    num_tasks: int,
    inner_lr: float,
    meta_lr: float,
    eval_adapt_steps: int,
    device: torch.device,
    out_dir: Path,
):
    """Classic toy MAML loop for sine regression."""
    ensure_dir(out_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    meta_model = SineMLP().to(device)
    opt_meta = optim.Adam(meta_model.parameters(), lr=meta_lr)

    for it in range(episodes):
        opt_meta.zero_grad()
        # sample tasks
        for _ in range(num_tasks):
            amp = random.uniform(0.1, 5.0)
            ph = random.uniform(0, math.pi)

            tmp = SineMLP().to(device)
            tmp.load_state_dict(meta_model.state_dict())
            inner_opt = optim.SGD(tmp.parameters(), lr=inner_lr)

            for _ in range(inner_steps):
                x_i, y_i = generate_sine(amp, ph, device=device)
                loss_i = mse(tmp(x_i), y_i)
                inner_opt.zero_grad()
                loss_i.backward()
                inner_opt.step()

            x_q, y_q = generate_sine(amp, ph, device=device)
            mse(tmp(x_q), y_q).backward()

        opt_meta.step()

    # Adaptation eval on a fixed task
    adapt = SineMLP().to(device)
    adapt.load_state_dict(meta_model.state_dict())
    opt_adapt = optim.SGD(adapt.parameters(), lr=inner_lr)

    adapt_curve = []
    for step in range(eval_adapt_steps):
        x, y = generate_sine(1.5, 0.5, device=device)
        loss = mse(adapt(x), y)
        adapt_curve.append(float(loss.item()))
        opt_adapt.zero_grad()
        loss.backward()
        opt_adapt.step()

    write_csv(out_dir / f"adapt_curve_seed{seed}.csv", [(i, v) for i, v in enumerate(adapt_curve)], header=["step", "mse"])
    return np.asarray(adapt_curve, dtype=np.float32)
