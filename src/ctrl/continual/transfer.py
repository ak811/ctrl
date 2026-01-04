from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import SineMLP
from .data import generate_sine
from ..common.io import write_csv, ensure_dir

mse = nn.MSELoss()

@dataclass
class TransferResult:
    curve: np.ndarray
    grad_curve: list[tuple[int, float]]

def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum().item())
    return float(total ** 0.5)

def run_transfer(src, tgt, seed: int, episodes: int, lr: float, log_interval: int, device: torch.device, out_dir: Path):
    (amp_s, ph_s), (amp_t, ph_t) = src, tgt
    tag = f"A{amp_s}_P{ph_s}_to_A{amp_t}_P{ph_t}_seed{seed}"
    ensure_dir(out_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train source model
    src_model = SineMLP().to(device)
    opt_s = optim.Adam(src_model.parameters(), lr=lr)
    for _ in range(episodes):
        x, y = generate_sine(amp_s, ph_s, device=device)
        loss = mse(src_model(x), y)
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
    torch.save(src_model.state_dict(), out_dir / f"src_{tag}.pth")

    results: dict[str, TransferResult] = {}

    for mode in ("scratch", "freeze", "finetune"):
        model = SineMLP().to(device)
        if mode != "scratch":
            model.load_state_dict(src_model.state_dict())
            if mode == "freeze":
                # Freeze first layer weights (simple heuristic)
                for name, p in model.named_parameters():
                    p.requires_grad = ("net.2" in name)  # last linear only

        opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

        curve = []
        grad_curve = []
        for ep in range(episodes):
            x, y = generate_sine(amp_t, ph_t, device=device)
            loss = mse(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            curve.append(float(loss.item()))
            if ep % log_interval == 0 or ep == episodes - 1:
                grad_curve.append((ep, grad_norm(model)))

        curve_arr = np.asarray(curve, dtype=np.float32)
        results[mode] = TransferResult(curve=curve_arr, grad_curve=grad_curve)

        write_csv(out_dir / f"{mode}_{tag}.csv", [(i, v) for i, v in enumerate(curve)], header=["episode", "mse"])
        write_csv(out_dir / f"{mode}_grad_{tag}.csv", grad_curve, header=["episode", "grad_norm"])

    return results
