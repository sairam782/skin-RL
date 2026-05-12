import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim=519, num_actions=7):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, state):
        shared = self.shared(state)
        value = self.value(shared)
        advantage = self.advantage(shared)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
