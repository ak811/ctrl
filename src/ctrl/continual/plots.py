from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .transfer import run_transfer
from .maml_sine import train_maml_sine
from .ewc import run_ewc
from .config import ContinualConfig
from ..common.io import ensure_dir

def _plot_with_band(xs, mean, std, label):
    plt.fill_between(xs, mean - std, mean + std, alpha=0.25)
    plt.plot(xs, mean, label=label)

def plot_transfer(cfg: ContinualConfig, device, out_root: Path):
    transfer_curves = {"scratch": [], "freeze": [], "finetune": []}
    out_dir = ensure_dir(out_root / "transfer")

    for seed in range(cfg.n_seeds):
        for pair in cfg.transfer_pairs:
            res = run_transfer(pair[0], pair[1], seed, cfg.episodes, cfg.learning_rate, cfg.log_interval, device, out_dir)
            for mode in transfer_curves:
                transfer_curves[mode].append(res[mode].curve)

    x = np.arange(cfg.episodes)
    plt.figure(figsize=(9, 5))
    for mode, curves in transfer_curves.items():
        arr = np.stack(curves, axis=0)
        _plot_with_band(x, arr.mean(0), arr.std(0), mode.capitalize())
    plt.title("Transfer Strategies (Sine Tasks)")
    plt.xlabel("Episode")
    plt.ylabel("MSE (lower is better)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png")
    plt.close()

def plot_maml(cfg: ContinualConfig, device, out_root: Path):
    out_dir = ensure_dir(out_root / "maml")
    curves = []
    for seed in range(cfg.n_seeds):
        curves.append(train_maml_sine(
            seed=seed,
            episodes=cfg.episodes,
            inner_steps=cfg.inner_steps,
            num_tasks=cfg.maml_num_tasks,
            inner_lr=cfg.maml_inner_lr,
            meta_lr=cfg.maml_meta_lr,
            eval_adapt_steps=cfg.eval_adapt_steps,
            device=device,
            out_dir=out_dir
        ))
    arr = np.stack(curves, axis=0)
    xs = np.arange(cfg.eval_adapt_steps)
    plt.figure(figsize=(9, 5))
    _plot_with_band(xs, arr.mean(0), arr.std(0), "MAML")
    plt.title("MAML Adaptation (Sine Task)")
    plt.xlabel("Adaptation step")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "meta_adaptation.png")
    plt.close()

def plot_ewc(cfg: ContinualConfig, device, out_root: Path):
    out_dir = ensure_dir(out_root / "ewc")
    fmats = []
    for seed in range(cfg.n_seeds):
        fmats.append(run_ewc(seed, cfg.task_params, cfg.episodes, cfg.learning_rate, cfg.ewc_lambda, device, out_dir))
    mean_fmat = np.stack(fmats, axis=0).mean(0)

    plt.figure(figsize=(7, 6))
    plt.imshow(mean_fmat, origin="lower", aspect="auto")
    plt.colorbar(label="MSE")
    ticks = np.arange(len(cfg.task_params))
    plt.xticks(ticks, [f"T{i+1}" for i in ticks], rotation=45)
    plt.yticks(ticks, [f"S{i+1}" for i in ticks])
    plt.title("EWC Forgetting Matrix (mean over seeds)")
    plt.tight_layout()
    plt.savefig(out_dir / "forgetting.png")
    plt.close()
