
"""

This is a sample implementation for the models presented
in a recent contribution: https://link.springer.com/article/10.1007/s10994-021-06106-3

"""

import torch.nn as nn
from deep_nilmtk.models.pytorch import S2P
from .layers import Attention

class SAED(S2P):
    def __init__(self, params):
        super(SAED, self).__init__(params)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 4, stride=1),
            nn.ReLU())
        self.att = Attention(128, attention_type=params['attention_type'])
        self.gru = nn.GRU(input_size=params['in_size'] - 3, hidden_size=64, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Tanh(),
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        if x.ndim != 3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x, hn = self.gru(x)
        x, weights = self.att(x, x)
        x = self.decoder(x)
        return x