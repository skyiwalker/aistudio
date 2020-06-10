from __future__ import print_function
import argparse
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

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
        return F.log_softmax(x)

def train(epoch):
    model.train()
    ##### HOROVOD ##### --- train_sampler, optimizer wrapper
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            ##### HOROVOD ##### --- train_sampler
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def metric_average(val, name):
    tensor = torch.tensor(val)
    ##### HOROVOD ##### -- allreduce (tensor average)
    avg_tensor = hvd.allreduce(tensor, name=name)    
    return avg_tensor.item()

def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    ##### HOROVOD #####
    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    ##### HOROVOD #####
    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    ##### HOROVOD #####
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

if __name__ == '__main__':
    
    ##### TODO: This is arguments set #####
    ##### CUDA SETTING #####
    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    #########################
    batch_size = 64
    test_batch_size = 128
    epochs = 5    
    momentum = 0.5
    lr = 0.01
    log_interval = 10
    # For Parallel/Distributed
    seed = 42
    use_adasum = False

    # Horovod: initialize library.
    ##### HOROVOD #####
    hvd.init()
    torch.manual_seed(seed)

    if cuda:
        # Horovod: pin GPU to local rank.
        ##### HOROVOD #####
        torch.cuda.set_device(hvd.local_rank())
        ##### TODO:Need argument #####
        #torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed(seed)


    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    ##### HOROVOD #####
    train_dataset = \
        datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    ##### HOROVOD #####
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    ##### TODO:Need argument #####
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
    ##### HOROVOD #####
    test_dataset = \
        datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    ##### HOROVOD #####
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    ##### TODO:Need argument #####
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                              sampler=test_sampler, **kwargs)

    model = Net()

    ##### HOROVOD #####
    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not use_adasum else 1

    if cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        ##### TODO:Need argument #####
        if use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    ##### TODO:Need argument #####
    '''
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                          momentum=args.momentum)'''
    optimizer = optim.SGD(model.parameters(), lr=lr * lr_scaler,
                          momentum=momentum)

    ##### HOROVOD #####
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    ##### HOROVOD #####
    # Horovod: (optional) compression algorithm.
    # compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    compression = hvd.Compression.none

    ##### HOROVOD #####
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if use_adasum else hvd.Average)

    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
