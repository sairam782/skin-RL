import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, feature_dim=512, prob_dim=7, num_actions=7):
        super().__init__()
        self.feature_dim = feature_dim

        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.final = nn.Sequential(
            nn.Linear(256 + prob_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state):
        features = state[:, : self.feature_dim]
        probs = state[:, self.feature_dim :]

        feature_embedding = self.feature_net(features)
        x = torch.cat([feature_embedding, probs], dim=1)
        return self.final(x)
