from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class PlottingCallback(BaseCallback):
    """Collect episode rewards and plot at the end."""
    def __init__(self, out_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 'infos' is a list for VecEnv; Monitor adds 'episode' dict at episode end
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None and "r" in ep:
                self.episode_rewards.append(float(ep["r"]))
        return True

    def _on_training_end(self) -> None:
        if not self.episode_rewards:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(self.episode_rewards, dtype=np.float32)

        plt.figure(figsize=(10, 5))
        plt.plot(arr, alpha=0.4, label="episode reward")
        if len(arr) >= 50:
            k = 50
            smooth = np.convolve(arr, np.ones(k) / k, mode="valid")
            plt.plot(np.arange(len(smooth)) + k - 1, smooth, label=f"moving avg ({k})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "reward_curve.png")
        plt.close()
