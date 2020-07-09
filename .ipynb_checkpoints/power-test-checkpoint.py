from aistudio.torchestimator import TorchEstimator

# USER-DEFINED
modulename = "mnist-linear"
######### DO NOT CHANGE #########
net_filename = modulename + ".py"
net_filename = "./nets/" + net_filename
#################################

# dataset_name(USER-DEFINED) - Feel free to change
dataset_name = "MNIST"
######### DO NOT CHANGE #########
dataset_filename = dataset_name + ".py"
dataset_filename = "./datasets/" + dataset_filename
#################################
    
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
    'loss':'cross_entropy',
    #'loss':'nll_loss',
    'optimizer':'SGD',
    'debug': True
}

estimator = TorchEstimator(net_name=modulename,script_params=script_params)
estimator.fit(dataset_loader=dataset_name)