from __future__ import annotations
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PatchEmbedCNN(nn.Module):
    def __init__(self, input_channels: int = 1, output_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class GeneralizedExtractor(BaseFeaturesExtractor):
    """Feature extractor shared across games (image obs: (C,H,W))."""
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.encoder = PatchEmbedCNN(input_channels=c, output_dim=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)
