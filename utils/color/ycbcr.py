#!/usr/bin/env python
import torch

class RGBToYCbCr(torch.nn.Module):
    """Convert RGB coordinates to YCbCr.
    
    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor, where C represents the RGB coordinates 
        of the image."""
    def __init__(self):
        super().__init__()
        weight = torch.tensor([[0.299,0.587,0.114],
                               [-0.169, -0.331, 0.5],
                               [0.5, -0.419, -0.081]]).transpose(0,1)
        bias = torch.tensor([0,0.5,0.5])
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, im):
        batch = im.shape[0]
        h = im.shape[2]
        w = im.shape[3]
        assert im.shape[1] == 3
        temp = im.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)
        result = torch.matmul(temp,self.weight) + self.bias.unsqueeze(0)
        return result.reshape(batch, h, w, 3).permute(0,3,1,2)

class YCbCrToRGB(torch.nn.Module):
    """Convert YCbCr coordinates to RGB.
    
    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor, where C represents the YCbCr 
        coordinates of the image."""
    def __init__(self):
        super().__init__()
        weight = torch.tensor([[0.299,0.587,0.114],
                               [-0.169, -0.331, 0.5],
                               [0.5, -0.419, -0.081]]).transpose(0,1)
        bias = torch.tensor([0,0.5,0.5])
        self.register_buffer('weight', torch.inverse(weight))
        self.register_buffer('bias', -1*bias)

    def forward(self, im):
        batch = im.shape[0]
        h = im.shape[2]
        w = im.shape[3]
        assert im.shape[1] == 3
        temp = im.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)
        result = torch.matmul(temp + self.bias.unsqueeze(0), self.weight)
        return result.reshape(batch, h, w, 3).permute(0,3,1,2)
