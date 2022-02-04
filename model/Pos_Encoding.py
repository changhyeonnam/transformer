import torch
import torch.nn as nn

class PositionwiseFFLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 pff_dim,
                 dropout):
        super(PositionwiseFFLayer, self).__init__()
        # original paper user hid_dim = 512, pff_dim = 2048
        self.fc_1 = nn.Linear(hid_dim, pff_dim)
        self.fc_2 = nn.Linear(pff_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x