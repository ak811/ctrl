from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import deque

class PongStackEnv(gym.Env):
    """Atari Pong wrapper that returns a stack of grayscale 40x40 frames (channels-first).

    This is used in the meta-learning module where we want a small CNN.
    Actions are mapped to Atari actions: [0 NOOP, 2 UP, 3 DOWN].
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, frame_stack: int = 6, render_mode: str | None = "rgb_array"):
        super().__init__()
        self.env = gym.make("ALE/Pong-v5", render_mode=render_mode)
        self.action_space = spaces.Discrete(3)
        self.frame_stack = int(frame_stack)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.frame_stack, 40, 40), dtype=np.float32)
        self._buf = deque(maxlen=self.frame_stack)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        frame = self._preprocess(obs)
        self._buf.clear()
        for _ in range(self.frame_stack):
            self._buf.append(frame)
        return np.stack(self._buf, axis=0), info

    def step(self, action: int):
        atari_action = [0, 2, 3][int(action)]
        obs, reward, terminated, truncated, info = self.env.step(atari_action)
        frame = self._preprocess(obs)
        self._buf.append(frame)
        return np.stack(self._buf, axis=0), float(reward), terminated, truncated, info

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (40, 40), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32) / 255.0
        return gray

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class PongImageEnv(gym.Env):
    """Pong wrapper producing (1,84,84) uint8 frames for SB3 transfer learning."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = "rgb_array"):
        super().__init__()
        self.env = gym.make("ALE/Pong-v5", render_mode=render_mode)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._preprocess(obs), info

    def step(self, action: int):
        atari_action = [0, 2, 3][int(action)]
        obs, reward, terminated, truncated, info = self.env.step(atari_action)
        return self._preprocess(obs), float(reward), terminated, truncated, info

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return gray[None, :, :].astype(np.uint8)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
