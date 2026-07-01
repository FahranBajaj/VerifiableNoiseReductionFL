import torch

class Model(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()
    
def loss():
    """
    Returns appropriate loss function for model
    e.g. return torch.nn.CrossEntropyLoss()
    """
    raise NotImplementedError()