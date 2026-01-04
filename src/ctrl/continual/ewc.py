from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import SineMLP
from .data import generate_sine
from ..common.io import write_csv, ensure_dir

mse = nn.MSELoss()

def compute_fisher(model: torch.nn.Module, loader):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for x_batch, y_batch in loader:
        model.zero_grad()
        loss = mse(model(x_batch), y_batch)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= max(1, len(loader))
    return fisher

@dataclass
class EWC:
    params: dict
    fisher: dict
    lam: float

    @classmethod
    def from_model(cls, model: torch.nn.Module, loader, lam: float):
        params = {n: p.clone().detach() for n, p in model.named_parameters()}
        fisher = compute_fisher(model, loader)
        return cls(params=params, fisher=fisher, lam=lam)

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        loss = 0.0
        for n, p in model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return (self.lam / 2.0) * loss

def run_ewc(seed: int, task_params, episodes: int, lr: float, lam: float, device: torch.device, out_dir: Path):
    ensure_dir(out_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = SineMLP().to(device)
    ewc_obj = None
    forgetting = np.zeros((len(task_params), len(task_params)), dtype=np.float32)

    for i, (amp, ph) in enumerate(task_params):
        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(episodes):
            x, y = generate_sine(amp, ph, device=device)
            loss = mse(model(x), y)
            if ewc_obj is not None:
                loss = loss + ewc_obj.penalty(model)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # build loader for fisher
        xs, ys = generate_sine(amp, ph, n=100, device=device)
        loader = [(xs[k:k+10], ys[k:k+10]) for k in range(0, 100, 10)]
        ewc_obj = EWC.from_model(model, loader, lam=lam)

        # evaluate on seen tasks
        for j, (amp_j, ph_j) in enumerate(task_params[: i + 1]):
            x_t, y_t = generate_sine(amp_j, ph_j, n=100, device=device)
            forgetting[i, j] = float(mse(model(x_t), y_t).item())

    write_csv(out_dir / f"forgetting_seed{seed}.csv", forgetting.tolist())
    return forgetting
