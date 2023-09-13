#!/usr/bin/env python
import torch

class CIEXYZToCIELAB(torch.nn.Module):
    '''Convert CIE 1931 standard tristimulus values (CIEXYZ, ISO/CIE 11664-1) to CIE 
    1976 L*a*b* values (CIELAB, ISO/CIE 11664-4:2019(E)) using standard illuminant D65.

    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor where the C channel represents the 
        CIEXYZ coordinates of the image.'''
    def __init__(self, xyz_n=None):
        super().__init__()
        if xyz_n is None:
            # Standard Illuminant D65
            self.register_buffer('xn', torch.tensor([95.047]))
            self.register_buffer('yn', torch.tensor([100]))
            self.register_buffer('zn', torch.tensor([108.883]))
        else:
            self.register_buffer('xn', torch.tensor(xyz_n[0]))
            self.register_buffer('yn', torch.tensor(xyz_n[1]))
            self.register_buffer('zn', torch.tensor(xyz_n[2]))

        self.register_buffer('delta3', torch.tensor([(6/29)**3]))
        self.register_buffer('delta2', torch.tensor([(6/29)**2]))

    def f(self, t):
        t = t.clone()
        mask = (t > self.delta3)
        t[mask] = t[mask]**(1/3)
        t[~mask] = t[~mask]/(3*self.delta2)+4/29
        return t

    def forward(self, im):
        X = im[:,0]*100
        Y = im[:,1]*100
        Z = im[:,2]*100
        L = 116*self.f(Y/self.yn) - 16
        a = 500*(self.f(X/self.xn)-self.f(Y/self.yn))
        b = 200*(self.f(Y/self.yn)-self.f(Z/self.zn))
        return torch.stack((L, a, b), dim=1)

class CIELABToCIEXYZ(torch.nn.Module):
    '''Convert CIE 1976 L*a*b* values (CIELAB, ISO/CIE 11664-4:2019(E)) to CIE 1931 
    standard tristimulus values (CIEXYZ, ISO/CIE 11664-1) using standard illuminant D65.
    
    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor where the C channel represents the 
        CIELAB coordinates of the image.'''
    def __init__(self):
        super().__init__()
        # Standard Illuminant D65
        self.register_buffer('xn', torch.tensor([95.047]))
        self.register_buffer('yn', torch.tensor([100]))
        self.register_buffer('zn', torch.tensor([108.883]))

        self.register_buffer('delta', torch.tensor([6/29]))

    def f_inv(self, t):
        t = t.clone()
        mask = (t > self.delta)
        t[mask] = t[mask]**3
        t[~mask] = 3*(self.delta**2)*(t[~mask] - 4/29)
        return t

    def forward(self, im):
        L = im[:,0]
        a = im[:,1]
        b = im[:,2]
        temp = (L+16)/116
        X = self.xn*self.f_inv(temp + a/500)
        Y = self.yn*self.f_inv(temp)
        Z = self.zn*self.f_inv(temp - b/200)
        return torch.stack((X/100,Y/100,Z/100), dim=1)
