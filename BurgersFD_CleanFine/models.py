"""
Pytorch neural network models
"""

from torch import nn
import pdb


class RNM_NN(nn.Module):
    def __init__(self, q1_size, q2_size):
        super(RNM_NN, self).__init__()
        self.elu_stack = nn.Sequential(
            nn.Linear(q1_size, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, q2_size)

            # nn.Linear(q1_size, int(pow(q2_size*q1_size,1/2))),
            # nn.ELU(),
            # nn.Linear(int(pow(q2_size*q1_size,1/2)), int(pow(q2_size*q1_size,3/4))),
            # nn.ELU(),
            # nn.Linear(int(pow(q2_size*q1_size,3/4)), int(pow(q2_size*q1_size, 4/5))),
            # nn.ELU(),
            # nn.Linear(int(pow(q2_size*q1_size, 4/5)), q2_size)
        )

    def forward(self, z):
        xhu = self.elu_stack(z)
        return xhu