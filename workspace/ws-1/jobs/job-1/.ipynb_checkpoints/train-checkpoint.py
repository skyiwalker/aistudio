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
from torchvision import datasets, models, transforms
# To Use Horovod
import torch.multiprocessing as mp
import torch.utils.data.distributed
import horovod.torch as hvd
#from torch.utils.tensorboard import SummaryWriter

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def log_epoch(phase_str, epoch, num_epochs, loss, acc):
    # global variables
    global this_path
    
    print('-' * 10)
    print('Phase: {}'.format(phase_str))
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('Loss: {:.4f} Acc: {:.4f}'.format(
          loss, acc*100.))
    print('-' * 10)    
#     writer.add_scalar('Loss/epoch', loss, epoch)
#     writer.add_scalar('Accuracy/epoch', acc, epoch)
    if phase_str != "Prediction":
        with open(this_path + '/epochs.log', 'a') as f:
            f.write("{}, {}, {}, {}\n".format(phase_str, epoch+1,loss,acc))
            f.flush()

def train(phase, epoch, num_epochs, model, loss_func, optimizer, dataloader, sampler=None):    
    # global variables
    global global_iteration, this_path
    
    if phase == "train":
        model.train()
    else:
        model.eval()
    
    phase_str = ""
    if phase == "train":
        phase_str = "Train"
    elif phase == "valid":
        phase_str = "Validation"
    elif phase == "test":
        phase_str = "Prediction"

    # Horovod: set epoch to sampler for shuffling.
    if cuda:
        sampler.set_epoch(epoch)
    # initialize running loss and accuracy    
    running_loss = 0.0
    running_accuracy = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        if cuda:
            data, target = data.cuda(), target.cuda()                    
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)                    
        acc = accuracy(output,target)
        if phase == "train":
            loss.backward()
            optimizer.step()
        # statistics
        running_loss += loss.item() * data.size(0)
        running_accuracy += acc * data.size(0)
        if batch_idx % args.log_interval == 0:
            if cuda:
                if hvd.rank() == 0:                    
                    print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                          phase_str, epoch+1, batch_idx * len(data), len(sampler),
                          100. * batch_idx / len(dataloader), loss.item(), acc*100.))
#                     writer.add_scalar('Loss/iteration', loss.item(), global_iteration)
#                     writer.add_scalar('Accuracy/iteration', acc, global_iteration)
                    if phase != "test":
                        with open(this_path + '/iterations.log', 'a') as f:
                            f.write("{}, {}, {}, {}\n".format(phase_str,global_iteration,loss.item(),acc))
                            f.flush()
            else:
                print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                      phase_str, epoch+1, batch_idx * len(data), len(dataloader.dataset),
                      100. * batch_idx / len(dataloader), loss.item(), acc*100.))
#                 writer.add_scalar('Loss/iteration', loss.item(), global_iteration)
#                 writer.add_scalar('Accuracy/iteration', acc, global_iteration)
                if phase != "test":
                    with open(this_path + '/iterations.log', 'a') as f:
                        f.write("{}, {}, {}, {}\n".format(phase_str,global_iteration,loss.item(),acc))
                        f.flush()
        global_iteration = global_iteration + 1
    # DataLoader Loop Ended.
    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_accuracy / dataset_size
    # print statistics for an epoch
    if cuda:
        if hvd.rank() == 0:
            log_epoch(phase_str, epoch, num_epochs, epoch_loss, epoch_acc)
    else:
        log_epoch(phase_str, epoch, num_epochs, epoch_loss, epoch_acc)
    
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
    parser.add_argument('--validation', action='store_true', default=False,
                        help='validation for training')
    parser.add_argument('--prediction', action='store_true', default=False,
                        help='predict for a model')
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
            
    # global variables
    global global_iteration, this_path
    global_iteration = 1
    
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
    this_path = os.path.dirname(os.path.abspath(__file__))
    # validation flag from validation argument
    validation = args.validation
    # prediction flag from prediction argument
    prediction = args.prediction
    
    # Dataset Loader
    if args.dataset_loader is not None:
        loader_name = args.dataset_loader
        # net path e.g.) $HOME/workspace/ws-1/jobs/job-1/../../../datasets
        dataset_path = os.path.join(this_path, os.pardir, os.pardir, 'datasets')        
        sys.path.append(dataset_path)
        # import dataset loader
        import importlib
        loader = importlib.import_module(loader_name)
        dataset_loader = loader.DatasetLoader()
        if prediction:
            test_dataset = dataset_loader.get_test_dataset()
        else:
            if validation:
                train_dataset, valid_dataset = dataset_loader.get_train_dataset(validation=True)
            else:
                train_dataset = dataset_loader.get_train_dataset(validation=False)
        
    else:    
        # Load Input Data        
        with open(this_path+'/dataset.pkl', 'rb') as f:
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

    # Handle Exception
    if args.model_path is not None and args.net_name is not None:
        print("Use only one of the model path and network.")
        # return main function
        sys.exit(0)
    
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
        netpath = os.path.join(this_path, os.pardir, os.pardir, 'nets')        
        sys.path.append(netpath)
        # Custom Model
        import importlib
        torchnet = importlib.import_module(modulename)
        # set model
        model = torchnet.Net()
        # load model to predict
        if prediction:
            model.load_state_dict(torch.load(this_path+"/torchmodel.pth"))
    
        
    if cuda:
        ##### HOROVOD #####
        if prediction:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                           test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                           train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            if validation:
                valid_sampler = torch.utils.data.distributed.DistributedSampler(
                           valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        kwargs = {'num_workers': 1, 'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
            
        if prediction:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                       sampler=test_sampler, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       sampler=train_sampler, **kwargs)
            if validation:
                valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                       sampler=valid_sampler, **kwargs)
        
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
        if prediction:
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            if validation:
                valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
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

                
    # Initialize Summary Writer
#     writer_train = SummaryWriter("runs/train")
#     writer_valid = SummaryWriter("runs/valid")
 
    # Initialize Global Iteration
    global_iteration = 0
    losses = []
    nums = []
    accs = []    
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        if cuda:
            if prediction:
                train("test",epoch,num_epochs,model,loss_func,optimizer,test_loader,test_sampler)
            else:
                train("train",epoch,num_epochs,model,loss_func,optimizer,train_loader,train_sampler)
                if validation:
                    train("valid",epoch,num_epochs,model,loss_func,optimizer,valid_loader,valid_sampler)
        else:
            if prediction:
                train("test",epoch,num_epochs,model,loss_func,optimizer,test_loader)
            else:
                train("train",epoch,num_epochs,model,loss_func,optimizer,train_loader)
                if validation:
                    train("valid",epoch,num_epochs,model,loss_func,optimizer,valid_loader)
          
    # save trained model
    if not prediction:
        PATH = this_path + '/torchmodel.pth'
        torch.save(model.state_dict(), PATH)
