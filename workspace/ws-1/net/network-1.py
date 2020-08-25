import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net_layers = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                (3, 3),
                dilation=(1, 1),
                padding=(1, 1),
                stride=(1, 1),
                padding_mode="zeros",
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(0, 0), stride=(2, 2)),
            nn.Conv2d(
                32,
                64,
                (3, 3),
                dilation=(1, 1),
                padding=(1, 1),
                stride=(1, 1),
                padding_mode="zeros",
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(0, 0), stride=(2, 2)),
            nn.Conv2d(
                64,
                128,
                (3, 3),
                dilation=(1, 1),
                padding=(1, 1),
                stride=(1, 1),
                padding_mode="zeros",
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1), stride=(2, 2)),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(2048, 625), nn.ReLU(), nn.Linear(625, 10)
        )

    def forward(self, x):
        x = self.net_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
