from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

class PuckWorldVectorEnv(gym.Env):
    """PuckWorld with vector observation (pos, vel, target)."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_steps: int = 300, render_mode: str | None = None, dt: float = 0.1):
        super().__init__()
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.dt = float(dt)

        self.action_space = spaces.Discrete(4)  # N/S/W/E thrust
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.position = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.position = self._rng.random(2).astype(np.float32)
        self.target = self._rng.random(2).astype(np.float32)
        self.velocity[:] = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity, self.target]).astype(np.float32)

    def step(self, action: int):
        self.steps += 1

        force = np.zeros(2, dtype=np.float32)
        if action == 0:
            force[1] = -0.1
        elif action == 1:
            force[1] = 0.1
        elif action == 2:
            force[0] = -0.1
        elif action == 3:
            force[0] = 0.1

        prev_dist = float(np.linalg.norm(self.position - self.target))
        self.velocity += self.dt * force
        self.position += self.dt * self.velocity
        self.position = np.clip(self.position, 0.0, 1.0)
        self.velocity *= 0.9
        new_dist = float(np.linalg.norm(self.position - self.target))

        reward = (prev_dist - new_dist) * 2.0
        terminated = new_dist < 0.05
        if terminated:
            reward += 5.0

        truncated = self.steps >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, {"distance": new_dist}

    def render(self):
        size = 120
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        px, py = (self.position * size).astype(int)
        tx, ty = (self.target * size).astype(int)
        cv2.circle(img, (px, py), 6, (255, 0, 0), -1)
        cv2.circle(img, (tx, ty), 4, (0, 0, 255), -1)
        return img


class PuckWorldImageEnv(gym.Env):
    """PuckWorld that returns an 84x84 grayscale image (channels-first) for SB3."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_steps: int = 300, render_mode: str | None = None):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self._rng = np.random.default_rng()

        self.puck = np.array([42.0, 42.0], dtype=np.float32)
        self.goal = np.array([42.0, 42.0], dtype=np.float32)
        self.steps = 0
        self.cumulative_distance = 0.0
        self._frame = np.zeros((84, 84, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.puck[:] = 42.0
        self.goal = self._rng.uniform(low=10, high=74, size=(2,)).astype(np.float32)
        self.steps = 0
        self.cumulative_distance = 0.0
        self._draw()
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.puck += action * 2.0
        self.puck = np.clip(self.puck, 0, 83)

        dist = float(np.linalg.norm(self.puck - self.goal))
        self.cumulative_distance += dist

        terminated = dist < 5.0
        truncated = self.steps >= self.max_steps

        # reward only at end, as in your original version
        if terminated or truncated:
            reward = -(self.cumulative_distance / max(self.steps, 1))
        else:
            reward = 0.0

        self._draw()
        return self._obs(), float(reward), terminated, truncated, {"distance": dist}

    def _draw(self):
        self._frame[:] = 0
        cv2.circle(self._frame, tuple(self.goal.astype(int)), 3, (0, 0, 255), -1)
        cv2.circle(self._frame, tuple(self.puck.astype(int)), 4, (0, 255, 0), -1)

    def _obs(self):
        gray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2GRAY)
        return gray[None, :, :].astype(np.uint8)

    def render(self):
        return self._frame.copy()
