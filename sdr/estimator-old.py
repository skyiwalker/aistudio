import subprocess
import torch
import torch.nn.functional as F
from torch import optim
import onnx
import onnxruntime
import numpy as np
from torch.utils.data import DataLoader

class Estimator:
    def __init__(self,script="",script_params={},directory="",use_gpu=False,nprocs=1,model="",use_model=False):
        self.script = script
        self.script_params = script_params
        self.directory = directory
        self.use_gpu = use_gpu
        self.nprocs = nprocs
        self.model = model
        self.use_model = use_model
    
    def to_numpy(self,tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def accuracy(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

    def test(self, loss_func, test_loader, ort_session):
        losses = []
        nums = []
        accs = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(data)}
                ort_outs = ort_session.run(None, ort_inputs)
                #output = model(data)
                output = torch.Tensor(ort_outs[0])
                loss = loss_func(output, target)
                acc = self.accuracy(output,target)
                losses.append(loss)
                accs.append(acc)
                nums.append(len(data))
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            val_acc = np.sum(np.multiply(accs, nums)) / np.sum(nums)
            print('Loss: {:.6f}\tAccuracy: {}'.format(val_loss, val_acc*100.))
        
    def fit(self):
        if not self.use_model:
            exec_script = self.script
            args = []
            for key, value in self.script_params.items():
                args.append(str(key))
                args.append(str(value))
            print(args)
            proc = subprocess.Popen(['python',exec_script,*args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = proc.communicate()
            print(out)    
    
    def validate(self, test_loader, loss=""):
        if self.use_model:
            # Load the ONNX model
            model_onnx = onnx.load(self.model)
            # Check that the IR is well formed
            onnx.checker.check_model(model_onnx)
            # Print a human readable representation of the graph
            onnx.helper.printable_graph(model_onnx.graph)
            ort_session = onnxruntime.InferenceSession(self.model)
            loss_func = F.mse_loss
            if loss == "nll_loss":
                loss_func = F.nll_loss
            if loss == "mse_loss":
                loss_func = F.mse_loss
            if loss == "cross_entropy":
                loss_func = F.cross_entropy
            if loss == "l1_loss":
                loss_func = F.l1_loss
            self.test(loss_func, test_loader, ort_session)

