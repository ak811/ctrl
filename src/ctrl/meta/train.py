from __future__ import annotations
import argparse
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from .config import MetaConfig
from .policy import PolicyNetwork
from .utils import rollout_episode
from .reptile import clone_model, reptile_update
from ..common.seed import set_seed
from ..envs.snake import SnakeGridEnv
from ..envs.puckworld import PuckWorldVectorEnv
from ..envs.pong import PongStackEnv

def main():
    parser = argparse.ArgumentParser(description="Meta-learning across Snake/PuckWorld/Pong (first-order Reptile style).")
    parser.add_argument("--iterations", type=int, default=MetaConfig.iterations)
    parser.add_argument("--k_shots", type=int, default=MetaConfig.k_shots)
    parser.add_argument("--inner_lr", type=float, default=MetaConfig.inner_lr)
    parser.add_argument("--meta_lr", type=float, default=MetaConfig.meta_lr)
    parser.add_argument("--seed", type=int, default=MetaConfig.seed)
    args = parser.parse_args()

    cfg = MetaConfig(iterations=args.iterations, k_shots=args.k_shots, inner_lr=args.inner_lr, meta_lr=args.meta_lr, seed=args.seed)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build envs
    envs = {
        "snake": SnakeGridEnv(),
        "puckworld": PuckWorldVectorEnv(),
        "pong": PongStackEnv(frame_stack=cfg.frame_stack),
    }

    # Separate meta-models per env (architectures differ). Still useful: learn good init per-env with fast inner updates.
    models = {}
    for name, env in envs.items():
        input_shape = env.observation_space.shape
        action_dim = env.action_space.n
        models[name] = PolicyNetwork(input_shape, action_dim, game_type=name, frame_stack=cfg.frame_stack).to(device)

    history = {k: {"meta_reward": [], "inner_reward": []} for k in envs.keys()}
    meta_losses = []

    for it in range(cfg.iterations):
        total_meta_reward = 0.0

        for name, env in envs.items():
            meta_model = models[name]
            adapted = clone_model(meta_model).to(device)
            inner_opt = optim.Adam(adapted.parameters(), lr=cfg.inner_lr)

            # Inner loop: k_shots episodes
            inner_rewards = []
            for _ in range(cfg.k_shots):
                loss, r, _steps = rollout_episode(env, adapted, device, cfg.gamma, cfg.max_steps_per_episode, cfg.entropy_bonus)
                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()
                inner_rewards.append(r)

            # Meta objective: evaluate adapted model on one more episode
            _, meta_r, _ = rollout_episode(env, adapted, device, cfg.gamma, cfg.max_steps_per_episode, cfg.entropy_bonus)
            total_meta_reward += meta_r

            history[name]["inner_reward"].append(float(np.mean(inner_rewards)))
            history[name]["meta_reward"].append(float(meta_r))

            # Reptile update (first-order)
            reptile_update(meta_model, adapted, cfg.meta_lr)

        meta_losses.append(-total_meta_reward)

        if it % 10 == 0 or it == cfg.iterations - 1:
            print(f"[meta] iter {it}/{cfg.iterations} | total_meta_reward={total_meta_reward:.2f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "meta" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    for name, m in models.items():
        torch.save(m.state_dict(), out_dir / f"{name}_meta_init.pth")

    # Plot rewards
    plt.figure(figsize=(10, 5))
    for name in history:
        plt.plot(history[name]["meta_reward"], label=f"{name} meta-reward")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Meta rewards (post-adaptation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "meta_rewards.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for name in history:
        plt.plot(history[name]["inner_reward"], label=f"{name} inner-reward (avg k_shots)")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Inner-loop rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "inner_rewards.png")
    plt.close()

    # Plot a proxy meta-loss
    plt.figure(figsize=(10, 5))
    plt.plot(meta_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Proxy loss (-total reward)")
    plt.title("Meta training proxy loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "meta_proxy_loss.png")
    plt.close()

    print(f"[meta] outputs written to: {out_dir}")

if __name__ == "__main__":
    main()
