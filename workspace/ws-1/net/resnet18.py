from torchvision import models
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):        
        super(Net, self).__init__()        
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model.fc = nn.Linear(num_ftrs, 2)        
        #self.add_module("resnet18", self.model)
    
    def forward(self, x):
        x = self.model(x)
        return x
