"""
Neural network architectures
"""

from torch import nn
import torch.nn.functional as F


class FCNNEncoder(nn.Module):
    """
    Defining the concept encoder for the synthetic dataset.
    """

    def __init__(self, num_inputs: int, num_hidden: int, num_deep: int):
        super(FCNNEncoder, self).__init__()

        self.fc0 = nn.Linear(num_inputs, num_hidden)
        self.bn0 = nn.BatchNorm1d(num_hidden)
        self.fcs = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_deep)])
        self.dp = nn.Dropout(0.05)

    def forward(self, x):
        z = self.bn0(self.dp(F.relu(self.fc0(x))))
        for bn, fc in zip(self.bns, self.fcs):
            z = bn(self.dp(F.relu(fc(z))))
        return z
