from __future__ import annotations
import argparse
import datetime
from pathlib import Path
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecVideoRecorder

from .config import TransferConfig
from .env_factory import make_env
from .extractor import GeneralizedExtractor
from .callbacks import PlottingCallback
from ..common.seed import set_seed

def main():
    parser = argparse.ArgumentParser(description="Transfer learning with PPO (Stable-Baselines3).")
    parser.add_argument("--env", choices=["pong", "snake", "puckworld"], default=TransferConfig.env)
    parser.add_argument("--timesteps", type=int, default=TransferConfig.timesteps)
    parser.add_argument("--seed", type=int, default=TransferConfig.seed)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()

    cfg = TransferConfig(env=args.env, timesteps=args.timesteps, seed=args.seed)
    set_seed(cfg.seed)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "transfer" / cfg.env / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dir = out_dir / "videos"
    env = DummyVecEnv([lambda: make_env(cfg.env, record=False)])
    # Our env returns (C,H,W) already, so transpose wrapper is not needed, but kept safe if swapped later.
    env = VecTransposeImage(env)

    if args.record_video:
        env = VecVideoRecorder(
            env,
            video_folder=str(video_dir),
            record_video_trigger=lambda step: step == 0,
            video_length=10_000,
            name_prefix=f"{cfg.env}_ppo"
        )

    policy_kwargs = dict(
        features_extractor_class=GeneralizedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 128],
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
        seed=cfg.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=policy_kwargs,
    )

    callback = PlottingCallback(out_dir=out_dir)
    model.learn(total_timesteps=cfg.timesteps, callback=callback)

    model_path = out_dir / f"ppo_{cfg.env}.zip"
    model.save(str(model_path))
    print(f"[transfer] saved model to: {model_path}")
    print(f"[transfer] outputs written to: {out_dir}")

    env.close()

if __name__ == "__main__":
    main()
