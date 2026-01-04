from __future__ import annotations
import torch
import torch.nn as nn

class SineMLP(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
