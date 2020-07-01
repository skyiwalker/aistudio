import subprocess
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import onnx
import onnxruntime
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import sys
import pickle
from torchvision import datasets, transforms
# To Use Horovod
import torch.multiprocessing as mp
import torch.utils.data.distributed
import horovod.torch as hvd


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

if __name__ == '__main__':
    # Arguments Parsing
    parser = argparse.ArgumentParser(description='AIStudio Training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
#     parser.add_argument('--fp16-allreduce', action='store_true', default=False,
#                         help='use fp16 compression during allreduce')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    # dummy argument
    parser.add_argument('--nprocs', type=int, default=1, metavar='N',
                        help='number of processors(default: 1)')
    parser.add_argument('--loss', type=str, default='cross_entropy', metavar='L',
                        help='loss function (default: cross entropy)')
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                        help='optimizer (default: SGD)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    parser.add_argument('--model-path', type=str, metavar="M",
                        help='path of the model file')
    parser.add_argument('--net-name', type=str, metavar="S",
                        help='network module name')
    parser.add_argument('--dataset-loader', type=str, metavar="S",
                        help='dataset loader name')

    args = parser.parse_args()
    print(args)
    if args.debug:
        print("Arguments Parsing Finished.")
    # Parsing Finished
    cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.debug:
        if cuda:
            print("CUDA Supported!")
        else:
            print("CUDA Not Supported!")
    
    # manual seed
    torch.manual_seed(args.seed)
    if cuda:        
        # Horovod: initialize library.
        ##### HOROVOD #####
        hvd.init()        
        # Horovod: pin GPU to local rank.
        ##### HOROVOD #####
        torch.cuda.set_device(hvd.local_rank())            
        torch.cuda.manual_seed(args.seed)
        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)
    
    # Get path for this training script
    thispath = os.path.dirname(os.path.abspath(__file__))
    
    # Dataset Loader
    if args.dataset_loader is not None:
        loader_name = args.dataset_loader
        # net path e.g.) $HOME/workspace/ws-1/jobs/job-1/../../../datasets
        dataset_path = os.path.join(thispath, os.pardir, os.pardir, 'datasets')        
        sys.path.append(dataset_path)
        # import dataset loader
        import importlib
        loader = importlib.import_module(loader_name)
        dataset_loader = loader.DatasetLoader()
        train_dataset = dataset_loader.get_train_dataset(validation=False)
    else:    
        # Load Input Data        
        with open(thispath+'/dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        input_data_np, input_labels_np = dataset
        input_data = torch.from_numpy(input_data_np)
        input_labels = torch.from_numpy(input_labels_np)
        # Check Input Data
        if input_data is None or input_labels is None:
            print("Input Data Not Found.")
            sys.exit()    
        # Make TensorDataset and DataLoader for PyTorch
        train_dataset = TensorDataset(input_data, input_labels)
    
    
    # Handling Input of Loss Function
    loss = args.loss
    loss_func = F.nll_loss
    if loss == "nll_loss":
        loss_func = F.nll_loss
    elif loss == "mse_loss":
        loss_func = F.mse_loss
    elif loss == "cross_entropy":
        loss_func = F.cross_entropy
    elif loss == "l1_loss":
        loss_func = F.l1_loss
    #loss_func = nn.CrossEntropyLoss()

    # Load Model    
    if args.model_path is not None:
        print("Model path was found.")
        # set system path to load model
        model_path = args.model_path
        sys.path.append(model_path)
        # Custom Model
        import torchmodel
        # set model
        model = torchmodel.Net()
        model.load_state_dict(torch.load(model_path+"/torchmodel.pth"))
    
    # Load Network
    if args.net_name is not None:
        print("Network was found.")
        # set system path to load model
        modulename = args.net_name
        # net path e.g.) $HOME/workspace/ws-1/jobs/job-1/../../../nets
        netpath = os.path.join(thispath, os.pardir, os.pardir, 'nets')        
        sys.path.append(netpath)
        # Custom Model
        import importlib
        torchnet = importlib.import_module(modulename)
        # set model
        model = torchnet.Net()
    
        
    if cuda:
        ##### HOROVOD #####
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                       train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        kwargs = {'num_workers': 1, 'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   sampler=train_sampler, **kwargs)        
        
        # Move model to GPU.
        model.cuda()
        
        # By default, Adasum doesn't need scaling up learning rate.
        lr_scalar = hvd.size() if not args.use_adasum else 1
        
        if args.use_adasum and hvd.nccl_built():
            lr_scalar = hvd.local_size()
        
        if args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr*lr_scalar,
                                  momentum=args.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr*lr_scalar,
                                  momentum=args.momentum)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: (optional) compression algorithm.
        #compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        compression = hvd.Compression.none
        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             compression=compression,
                                             op=hvd.Average)
                                             #op=hvd.Adasum if args.use_adasum else hvd.Average)
    else:                
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        if args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum)
            

    if args.debug:
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    losses = []
    nums = []
    accs = []
    for epoch in range(args.epochs):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        if cuda:
            train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()                    
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)                    
            acc = accuracy(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                if cuda:
                    if hvd.rank() == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                              epoch+1, batch_idx * len(data), len(train_sampler),
                              100. * batch_idx / len(train_loader), loss.item(), acc*100))
                else:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                          epoch+1, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item(), acc*100))  
                    
    # save trained model
    PATH = thispath + '/torchmodel.pth'
    torch.save(model.state_dict(), PATH)