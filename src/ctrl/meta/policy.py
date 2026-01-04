from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Policy network:
    - Pong: small CNN over stacked 40x40 frames
    - Vector envs: MLP
    """
    def __init__(self, input_shape, action_dim: int, game_type: str, hidden_dims=(128, 64), conv_filters=(16, 32), dropout=0.1, frame_stack: int = 6):
        super().__init__()
        self.game_type = game_type
        self.action_dim = int(action_dim)

        self.dropout = nn.Dropout(dropout)

        if game_type == "pong":
            c = frame_stack
            self.conv1 = nn.Conv2d(c, conv_filters[0], kernel_size=4, stride=2)
            self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, stride=2)

            # 40x40 -> conv: (40-4)/2+1=19 -> (19-3)/2+1=9
            conv_out = conv_filters[1] * 9 * 9
            self.fc1 = nn.Linear(conv_out, hidden_dims[0])
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.head = nn.Linear(hidden_dims[1], self.action_dim)
        else:
            in_dim = int(input_shape[0])
            self.fc1 = nn.Linear(in_dim, hidden_dims[0])
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.head = nn.Linear(hidden_dims[1], self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.game_type == "pong":
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            logits = self.head(x)
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            logits = self.head(x)
        return logits
