from __future__ import annotations
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor

from ..envs.pong import PongImageEnv
from ..envs.snake import SnakeImageEnv
from ..envs.puckworld import PuckWorldImageEnv

def make_env(env_name: str, record: bool = False, video_dir: str = "videos"):
    env_name = env_name.lower().strip()
    if env_name == "pong":
        env = PongImageEnv(render_mode="rgb_array")
    elif env_name == "snake":
        env = SnakeImageEnv(render_mode="rgb_array")
    elif env_name == "puckworld":
        env = PuckWorldImageEnv(render_mode="rgb_array")
    else:
        raise ValueError(f"Unknown env '{env_name}'. Choose from: pong, snake, puckworld")

    if record:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)

    env = Monitor(env)
    return env
