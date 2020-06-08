from aistudio.estimator import Estimator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# To Use Horovod
import torch.multiprocessing as mp
import torch.utils.data.distributed
import horovod.torch as hvd
import subprocess
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os


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

class Estimator:
    def __init__(self,script="",script_params={},directory="",use_cuda=False,seed=42,use_adasum=False,nprocs=1,use_model=False,model_path="",network=None,use_optimizer=False,optimizer=None,debug=False):        
        self.use_cuda = use_cuda
        self.cuda = self.use_cuda and torch.cuda.is_available()
        self.script = script
        self.script_params = script_params
        self.directory = directory        
        self.seed = seed
        self.use_adasum = use_adasum
        self.nprocs = nprocs
        self.model_path = model_path
        self.use_model = use_model
        self.network = network
        self.use_optimizer = use_optimizer
        self.optimizer = optimizer
        self.debug = debug
        self.model_onnx = None
        self.model = None
        
        if self.cuda:            
            print("CUDA Supported!")
            # Horovod: initialize library.
            ##### HOROVOD #####
            hvd.init()            
            torch.manual_seed(seed)
            # Horovod: pin GPU to local rank.
            ##### HOROVOD #####
            torch.cuda.set_device(hvd.local_rank())            
            torch.cuda.manual_seed(seed)
            # Horovod: limit # of CPU threads to be used per worker.
            torch.set_num_threads(1)
                
        if self.use_model:
            self.model_filename, self.model_file_extension = os.path.splitext(self.model_path)            
            if self.model_file_extension == '.onnx':
                if self.cuda:
                    if hvd.rank() == 0:
                        print("ONNX Model was Loaded.")    
                else:
                    print("ONNX Model was Loaded.")
                # Load the ONNX model
                self.model_onnx = onnx.load(self.model_path)
                # Check that the IR is well formed
                onnx.checker.check_model(self.model_onnx)
                # Print a human readable representation of the graph
                onnx.helper.printable_graph(self.model_onnx.graph)
                ort_session = onnxruntime.InferenceSession(self.model_path)                
            elif self.model_file_extension == '.pth' or self.model_file_extension == '.pt':
                if self.cuda:
                    # print("PyTorch Model was Loaded:",hvd.rannk())
                    if hvd.rank() == 0:
                        print("PyTorch Model was Loaded.")
                else:
                    print("PyTorch Model was Loaded.")
                self.model = self.network
                # Only For Inference
                if self.use_optimizer:
                    # For Inference and Training
                    checkpoint = torch.load(self.model_path)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    self.model.load_state_dict(torch.load(self.model_path))
            
            if self.cuda:
                if self.model:
                    # Move model to GPU.
                    self.model.cuda()
                
            # set hyperparameters
            self.epochs = 5
            self.batch_size = 16
            self.test_batch_size = 32            
            self.lr = 0.01
            self.momentum = 0.5
            self.log_interval = 10
            if '--epochs' in self.script_params:
                self.epochs = self.script_params['--epochs']
            if '--batch-size' in self.script_params:
                self.batch_size = self.script_params['--batch-size']
            if '--test-batch-size' in self.script_params:
                self.test_batch_size = self.script_params['--test-batch-size']
            if '--lr' in self.script_params:
                self.lr = self.script_params['--lr']
            if '--momentum' in self.script_params:
                self.momentum = self.script_params['--momentum']
            if '--log-interval' in self.script_params:
                self.momentum = self.script_params['--log-interval']

    
    def to_numpy(self,tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def accuracy(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

    def test(self, loss_func, test_loader, model=None, ort_session=None):
        losses = []
        nums = []
        accs = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if self.model_onnx is not None:
                    ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(data)}
                    ort_outs = ort_session.run(None, ort_inputs)
                    #output = model(data)
                    output = torch.Tensor(ort_outs[0])
                elif self.model is not None:
                    output = self.model(data)
                loss = loss_func(output, target)
                acc = self.accuracy(output,target)
                losses.append(loss)
                accs.append(acc)
                nums.append(len(data))
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            val_acc = np.sum(np.multiply(accs, nums)) / np.sum(nums)
            print('Loss: {:.6f}\tAccuracy: {}'.format(val_loss, val_acc*100.))
        
    def fit(self, input_data=None, input_labels=None, loss="", opt=""):
        if self.use_model: # use_model
            # Check Input Data
            if input_data is None or input_labels is None:
                return
            if self.model_onnx:
                print("Cannot use onnx type to fit model")
                return
            # Make TensorDataset and DataLoader for PyTorch
            train_dataset = TensorDataset(input_data, input_labels)
            # Handling Input of Loss Function
            loss_func = F.nll_loss
            if loss == "nll_loss":
                loss_func = F.nll_loss
            elif loss == "mse_loss":
                loss_func = F.mse_loss
            elif loss == "cross_entropy":
                loss_func = F.cross_entropy
            elif loss == "l1_loss":
                loss_func = F.l1_loss
            if self.cuda:
                ##### HOROVOD #####
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                               train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
                kwargs = {'num_workers': 1, 'pin_memory': True}
                # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
                # issues with Infiniband implementations that are not fork-safe
                if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                        mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
                    kwargs['multiprocessing_context'] = 'forkserver'
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                           sampler=train_sampler, **kwargs)
                # Set Optimizer
                if self.use_optimizer:
                    optimizer = self.optimizer
                else:
                    if args.use_adasum and hvd.nccl_built():
                        lr_scaler = hvd.local_size()                
                    if opt == "SGD":
                        optimizer = optim.SGD(self.model.parameters(), lr=self.lr*lr_scalar,
                                              momentum=self.momentum)
                    else:
                        optimizer = optim.SGD(self.model.parameters(), lr=self.lr*lr_scalar,
                                              momentum=self.momentum)
                
                # Horovod: broadcast parameters & optimizer state.
                hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)
                
                # Horovod: (optional) compression algorithm.
                #compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
                compression = hvd.Compression.none
                # Horovod: wrap optimizer with DistributedOptimizer.
                optimizer = hvd.DistributedOptimizer(optimizer,
                                                     named_parameters=self.model.named_parameters(),
                                                     compression=compression,
                                                     op=hvd.Average)
                                                     #op=hvd.Adasum if args.use_adasum else hvd.Average)
            else:                
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
                if self.use_optimizer:
                    optimizer = self.optimizer
                else:
                    if optim == "SGD":
                        optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                              momentum=self.momentum)
                    else:
                        optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                              momentum=self.momentum)
            
            if self.debug:
                # Print model's state_dict
                print("Model's state_dict:")
                for param_tensor in self.model.state_dict():
                    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

                # Print optimizer's state_dict
                print("Optimizer's state_dict:")
                for var_name in optimizer.state_dict():
                    print(var_name, "\t", optimizer.state_dict()[var_name])
                
            losses = []
            nums = []
            accs = []
            for epoch in range(self.epochs):
                self.model.train()
                # Horovod: set epoch to sampler for shuffling.
                if self.cuda:
                    train_sampler.set_epoch(epoch)
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    if self.cuda:
                        data, target = data.cuda(), target.cuda()                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = loss_func(output, target)                    
                    acc = self.accuracy(output,target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % self.log_interval == 0:
                        if self.cuda:
                            if hvd.rank() == 0:
                                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                                      epoch+1, batch_idx * len(data), len(train_sampler),
                                      100. * batch_idx / len(train_loader), loss.item(), acc*100))
                        else:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                                  epoch+1, batch_idx * len(data), len(train_loader.dataset),
                                  100. * batch_idx / len(train_loader), loss.item(), acc*100))                                  

                            
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
    estimator = Estimator(use_cuda=True,use_model=True,model_path="mnist_init_net_checkpoint.pt",network=net,use_optimizer=True,optimizer=optimizer,script_params=script_params)
    estimator.fit(input_data, input_labels, loss="nll_loss", opt="SGD")
