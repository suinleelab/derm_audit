#!/usr/bin/env python
import torch
from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -1*x.mean()

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, real, fake):
        return torch.mean(self.relu(fake+1) + self.relu(-1*real+1))

class KLLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, true, pred):
        out = true*torch.log(pred+self.eps) + (1-true)*torch.log(1-pred+self.eps)
        return -1*out.mean()

