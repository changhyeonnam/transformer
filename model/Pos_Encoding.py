import torch
import torch.nn as nn

class PositionwiseFFLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 pf_dim,
                 dropout):

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)

        return x