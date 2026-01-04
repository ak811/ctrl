from __future__ import annotations
import argparse
from pathlib import Path
import datetime
import torch

from .config import ContinualConfig
from .plots import plot_transfer, plot_maml, plot_ewc

def main():
    parser = argparse.ArgumentParser(description="Continual learning experiments (sine tasks).")
    parser.add_argument("--all", action="store_true", help="Run all experiments and plots.")
    parser.add_argument("--transfer", action="store_true", help="Run transfer experiment.")
    parser.add_argument("--maml", action="store_true", help="Run MAML sine experiment.")
    parser.add_argument("--ewc", action="store_true", help="Run EWC forgetting experiment.")
    args = parser.parse_args()

    cfg = ContinualConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("outputs") / "continual" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    if args.all or (not args.transfer and not args.maml and not args.ewc):
        plot_transfer(cfg, device, out_root)
        plot_maml(cfg, device, out_root)
        plot_ewc(cfg, device, out_root)
    else:
        if args.transfer:
            plot_transfer(cfg, device, out_root)
        if args.maml:
            plot_maml(cfg, device, out_root)
        if args.ewc:
            plot_ewc(cfg, device, out_root)

    print(f"[continual] outputs written to: {out_root}")

if __name__ == "__main__":
    main()
