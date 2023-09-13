#!/usr/bin/env python
import torch

class CIEXYZToCIELUV(torch.nn.Module):
    '''Convert CIE 1931 standard tristimulus values (CIEXYZ, ISO/CIE 11664-1) to CIE 
    1976 L*u*v* (CIELUV, ISO/CIE 11664-5:2016) using the D65 standard illuminant.

    Inputs are assumed to be scaled such that the D65 standard illuminant is 
    represented as (0.95047, 1, 1.08883). In the output, L* ranges from 0 to 100.

    Note that black is represented as (100,nan,nan).
    
    Arguments:
    (forward)
      im: (torch.Tensor) An NCWH tensor where the C channel represents the
        CIEXYZ coordinates of the image.'''
    def __init__(self):
        super().__init__()
        x_d65 = 0.31271
        y_d65 = 0.32902
        u_prime_d65 = 4*x_d65/(-2*x_d65+12*y_d65+3)
        v_prime_d65 = 9*y_d65/(-2*x_d65+12*y_d65+3)

        # Maximum L* is 100
        self.register_buffer('Yn', torch.tensor([100]))
        self.register_buffer('un_prime', torch.tensor([u_prime_d65]))
        self.register_buffer('vn_prime', torch.tensor([v_prime_d65]))

    def _u_prime(self, X, Y, Z):
        u_prime = 4*X/(X+15*Y+3*Z)
        return u_prime

    def _v_prime(self, X, Y, Z):
        v_prime = 9*Y/(X+15*Y+3*Z)
        return v_prime

    def forward(self, im):
        X = im[:,0]*100
        Y = im[:,1]*100
        Z = im[:,2]*100

        u_prime = self._u_prime(X, Y, Z)
        v_prime = self._v_prime(X, Y, Z)

        mask = (Y/self.Yn <= (6/29)**3)
        L_star = Y.clone()
        L_star[mask] = (29/3)**3*Y[mask]/self.Yn
        L_star[~mask] = 116*(Y[~mask]/self.Yn)**(1/3)-16
        u_star = 13*L_star*(u_prime - self.un_prime)
        v_star = 13*L_star*(v_prime - self.vn_prime)
        return torch.stack((L_star, u_star, v_star), dim=1)

class CIELUVToCIEXYZ(torch.nn.Module):
    '''Convert CIE 1976 L*u*v* (CIELUV, ISO/CIE 11664-5:2016) to CIE 1931 standard 
    tristimulus values (CIEXYZ, ISO/CIE 11664-1) using the standard D65 illuminant.
    
    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor where the C channels represents the 
        CIELUV coordinates of the image.'''
    def __init__(self):
        super().__init__()
        x_d65 = 0.31271
        y_d65 = 0.32902
        u_prime_d65 = 4*x_d65/(-2*x_d65+12*y_d65+3)
        v_prime_d65 = 9*y_d65/(-2*x_d65+12*y_d65+3)

        # Maximum L* is 100
        self.register_buffer('Yn', torch.tensor([100]))
        self.register_buffer('un_prime', torch.tensor([u_prime_d65]))
        self.register_buffer('vn_prime', torch.tensor([v_prime_d65]))

    def forward(self, im):
        L_star = im[:,0]
        u_star = im[:,1]
        v_star = im[:,2]

        u_prime = u_star/(13*L_star)+self.un_prime
        v_prime = v_star/(13*L_star)+self.vn_prime
        mask = (L_star <= 8)
        Y = L_star.clone()
        Y[mask] = self.Yn*L_star[mask]*(3/29)**3
        Y[~mask] = self.Yn*((L_star[~mask]+16)/116)**3
        X = Y*9*u_prime/(4*v_prime)
        X = torch.nan_to_num(X, nan=0)
        Z = Y*(12-3*u_prime-20*v_prime)/(4*v_prime)
        Z = torch.nan_to_num(Z, nan=0)
        return torch.stack((X/100, Y/100, Z/100), dim=1)

class CIEXYZToCIELCh(CIEXYZToCIELUV):
    """Convert CIEXYZ coordinates to CIELCh (cylindrical CIELUV)..
    
    Arguments:
    (forward):
      im: (torch.Tensor) An NCHW tensor where the C channel represents the 
        CIEXYZ coordinates of the image."""
    def forward(self, im):
        luv = super().forward(im)
        l = luv[:,0]
        u = luv[:,1]
        v = luv[:,2]
        chroma = torch.sqrt(u**2+v**2)
        hue = torch.atan2(v, u)
        return torch.stack((l, chroma, hue), dim=1)

class CIELChToCIEXYZ(CIELUVToCIEXYZ):
    """Convert CIELCh (cylindrical CIELUV) to CIEXYZ coordinates.
    
    Arguments:
    (forward):
      im: (torch.Tensor) An NCHW tensor where the C channel represents the 
        CIELCh coordinates of the image."""
    def forward(self, im):
        l = im[:,0]
        chroma = im[:,1]
        hue = im[:,2]
        u = torch.cos(hue)*chroma
        v = torch.sin(hue)*chroma
        luv = torch.stack((l,u,v), dim=1)
        return super().forward(luv)
