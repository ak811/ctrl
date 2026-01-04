from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

class SnakeGridEnv(gym.Env):
    """A small Snake environment (grid-based) with vector observations.

    Observation: flattened grid (0 empty, 1 snake, 2 food)
    Actions: 0 up, 1 down, 2 left, 3 right
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, grid_size: int = 8, max_steps: int = 200, render_mode: str | None = None):
        super().__init__()
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.grid_size * self.grid_size,), dtype=np.float32)

        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] = (0, 0)
        self.steps = 0
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        center = (self.grid_size // 2, self.grid_size // 2)
        self.snake = [center]
        self.food = self._place_food()
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def _place_food(self) -> tuple[int, int]:
        while True:
            f = (int(self._rng.integers(0, self.grid_size)), int(self._rng.integers(0, self.grid_size)))
            if f not in self.snake:
                return f

    def _get_obs(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.snake:
            grid[x, y] = 1.0
        fx, fy = self.food
        grid[fx, fy] = 2.0
        return grid.flatten()

    def step(self, action: int):
        self.steps += 1
        head_x, head_y = self.snake[0]

        if action == 0:   # up
            head_x -= 1
        elif action == 1: # down
            head_x += 1
        elif action == 2: # left
            head_y -= 1
        elif action == 3: # right
            head_y += 1

        new_head = (head_x, head_y)

        terminated = False
        reward = 0.0

        # collision
        if (head_x < 0 or head_x >= self.grid_size or head_y < 0 or head_y >= self.grid_size or new_head in self.snake):
            terminated = True
            reward = -1.0
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 1.0
                self.food = self._place_food()
            else:
                self.snake.pop()
                # tiny shaping: encourage survival
                reward = 0.01

        truncated = self.steps >= self.max_steps
        obs = self._get_obs()
        info = {}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # Small RGB image for debugging
        img = np.zeros((self.grid_size * 20, self.grid_size * 20, 3), dtype=np.uint8)
        for x, y in self.snake:
            cv2.rectangle(img, (y * 20, x * 20), ((y+1) * 20, (x+1) * 20), (0, 255, 0), -1)
        fx, fy = self.food
        cv2.rectangle(img, (fy * 20, fx * 20), ((fy+1) * 20, (fx+1) * 20), (0, 0, 255), -1)
        return img


class SnakeImageEnv(gym.Env):
    """Snake environment that returns an 84x84 grayscale image (channels-first) for SB3 CNN policies."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, grid_size: int = 10, max_steps: int = 500, render_mode: str | None = None):
        super().__init__()
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

        self._core = SnakeGridEnv(grid_size=self.grid_size, max_steps=self.max_steps, render_mode=render_mode)

    def reset(self, seed=None, options=None):
        obs, info = self._core.reset(seed=seed, options=options)
        return self._to_img(), info

    def step(self, action: int):
        _, reward, terminated, truncated, info = self._core.step(action)
        return self._to_img(), reward, terminated, truncated, info

    def _to_img(self) -> np.ndarray:
        # Render a crisp grid then resize to 84x84
        img = self._core.render()  # RGB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)
        return gray[None, :, :].astype(np.uint8)

    def render(self):
        return self._core.render()
