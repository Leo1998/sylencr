
#!/usr/bin/env python3

import torch
from torch import nn

class CustomMaskMSE(nn.Module):
    def __init__(self, batch_size, alpha):
        super(CustomMaskMSE, self).__init__()
        self.batch_size = batch_size
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        upper = torch.relu(inputs - targets).square()
        lower = torch.relu(targets - inputs).square()

        return torch.sum(upper * self.alpha + lower) / self.batch_size
