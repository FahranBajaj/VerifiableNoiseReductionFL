import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Model (same one used in FedDMC)"""

    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(self.flatten(x)))
        return self.fc2(x)

    
def loss():
    """
    Returns appropriate loss function for model
    e.g. return torch.nn.CrossEntropyLoss()
    """
    return torch.nn.CrossEntropyLoss()