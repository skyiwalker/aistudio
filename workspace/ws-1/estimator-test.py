import sys
sys.path.append('../../')
from aistudio.estimator3 import Estimator

import torch
from torchvision import datasets, transforms

download_root = 'train-data'
mnist_transform=transforms.Compose([
                           transforms.ToTensor()
                           ,transforms.Normalize((0.1307,), (0.3081,))
                           ])
train_dataset  = datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True)
input_data_torch = train_dataset.data
input_data_torch = torch.div(input_data_torch,255.)
input_data_torch = torch.add(input_data_torch,-0.1307)
input_data_torch = torch.div(input_data_torch,0.3081)
input_data_torch = input_data_torch.unsqueeze(1)
input_labels_torch = train_dataset.targets
# Convert Tensor to Numpy
input_data = input_data_torch.numpy()
input_labels = input_labels_torch.numpy()

modulename = "torchnet"
script_params = {
    'epochs':5,
    'batch-size':64,
    'test-batch-size':128,
    'lr':0.01,
    'momentum':0.5,
    'seed':42,
    'log-interval':10,
    'no-cuda':False,
    'nprocs':1,
    #'loss':'cross_entropy',
    'loss':'nll_loss',
    'optimizer':'SGD',
    'debug': True
}
estimator = Estimator(net_name=modulename,script_params=script_params)
estimator.fit(input_data,input_labels)
