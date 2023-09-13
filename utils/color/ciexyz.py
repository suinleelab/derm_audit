#!/usr/bin/env python
import torch

def torchvision_to_unit_interval(x):
    return x/2+0.5

def unit_interval_to_torchvision(x):
    return x*2-1

def srgb_to_linear_rgb(x):
    '''Convert sRGB to linear RGB values (gamma transformation).'''
    x = x.clone()
    mask = (x <= 0.04045)
    x[mask] = x[mask]/12.92
    x[~mask] = ((x[~mask]+0.055)/1.055)**2.4
    return x

def linear_rgb_to_srgb(x):
    '''Reverse gamma correction to convert linear RGB values to sRGB values.'''
    x = torch.clamp(x, min=0, max=1)
    mask = (x <= 0.0031308)
    x[mask] = x[mask]*12.92
    x[~mask] = 1.055*x[~mask]**(1/2.4)-0.055
    return x

class SRGBToCIEXYZ(torch.nn.Module):
    '''Convert sRGB to CIE 1931 standard tristumulus values (CIEXYZ, ISO/CIE 11664-1). 
    Primaries and white point (encoded in matrix "w") based on Rec.709.

    Inputs are assumed to be in the range (0,1). The range of outputs is:
        0 <= X <= 0.95047
        0 <= Y <= 1
        0 <= Z <= 1.08883
    Note that the maximum values correspond to the coordinates of the standard 
    illuminant D65 used in sRGB.

    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor where the C channel represents the sRGB
        coordinates of the image.'''
    def __init__(self):
        super().__init__()
        w = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126728, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
        self.register_buffer("w", w.transpose(0,1))

    def forward(self, im):
        batch = im.shape[0]
        h = im.shape[2]
        w = im.shape[3]
        assert im.shape[1] == 3
        temp = im.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)
        temp = srgb_to_linear_rgb(temp)
        result = torch.matmul(temp, self.w)
        return result.reshape(batch, h, w, 3).permute(0,3,1,2)

class CIEXYZToSRGB(torch.nn.Module):
    '''Convert CIE 1931 standard tristumulus values (CIEXYZ, ISO/CIE 11664-1) to sRGB. 
    Primaries and white point (encoded in matrix "w") based on Rec.709.

    Inputs are assumed to be in the ranges:
        0 <= X <= 0.95047
        0 <= Y <= 1
        0 <= Z <= 1.08883
    Outputs are clamped to the range (0,1). 

    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor where the C channel represents the 
        CIEXYZ coordinates of the image.'''
    def __init__(self):
        super().__init__()

        w = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126728, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
        self.register_buffer("w_inv", torch.inverse(w.transpose(0,1)))

    def forward(self, im):
        batch = im.shape[0]
        h = im.shape[2]
        w = im.shape[3]
        assert im.shape[1] == 3
        temp = im.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)
        temp = torch.matmul(temp, self.w_inv)

        result = linear_rgb_to_srgb(temp)
        return result.reshape(batch, h, w, 3).permute(0,3,1,2)
