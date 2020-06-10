
from aistudio.estimator import Estimator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# import horovod.torch as hvd

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

if __name__ == '__main__':
    # define script parameters
    script_params = {
        '--test-batch-size': 128
    }
    # Data Preparation and Pre-Processing
    download_root = 'train-data'
    mnist_transform=transforms.Compose([
                               transforms.ToTensor()
                               ,transforms.Normalize((0.1307,), (0.3081,))
                               ])
    train_dataset  = datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True)
    input_data = train_dataset.data
    input_data = torch.div(input_data,255.)
    input_data = torch.add(input_data,-0.1307)
    input_data = torch.div(input_data,0.3081)
    input_data = input_data.unsqueeze(1)
    input_labels = train_dataset.targets

    script_params = {
        '--epochs': 5,
        '--batch-size': 64,
        '--test-batch-size': 128,
        '--lr': 0.05
    }
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.05)
    estimator = Estimator(use_cuda=True,use_model=True,model_path="mnist_init_net.pt",network=net,script_params=script_params)
    estimator.fit(input_data, input_labels, loss="nll_loss", opt="SGD")
