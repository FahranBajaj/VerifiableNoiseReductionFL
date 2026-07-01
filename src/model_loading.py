import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
        )

    def forward(self, x):
        return self.model(x)
    
def loss():
    return torch.nn.CrossEntropyLoss()