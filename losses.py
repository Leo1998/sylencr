
#!/usr/bin/env python3

import torch
from torch import nn

class CustomMaskMSE(nn.Module):
    def __init__(self, alpha):
        super(CustomMaskMSE, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        upper = torch.relu(inputs - targets).square()
        lower = torch.relu(targets - inputs).square()

        return torch.mean(upper * (1.0 - (1.0 / self.alpha)) + lower * (1 / self.alpha))
