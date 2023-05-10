#!/usr/bin/env python
from PIL import Image
import torch
import numpy as np
import colour

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

class CIEXYZToCIELAB(torch.nn.Module):
    '''Convert CIE 1931 standard tristimulus values (CIEXYZ, ISO/CIE 11664-1) to CIE 
    1976 L*a*b* values (CIELAB, ISO/CIE 11664-4:2019(E)) using standard illuminant D65.

    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor where the C channel represents the 
        CIEXYZ coordinates of the image.'''
    def __init__(self):
        super().__init__()
        # Standard Illuminant D65
        self.register_buffer('xn', torch.tensor([95.047]))
        self.register_buffer('yn', torch.tensor([100]))
        self.register_buffer('zn', torch.tensor([108.883]))

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

class BradfordAdaptation(torch.nn.Module):
    '''Transform images exposed in a test illuminant to appear as if they were 
    exposed in the reference illuminant, using the Bradford chromatic 
    adaptation transform (1,2). Chromatic adaptation is assumed to be complete (that 
    is, we take D=1.0 using Luo & Hunt's original notation.(2))

    References:
    (1) K.M. Lam, "Metamerism and colour constancy." PhD. Thesis, University of
      Bradford, 1985.
    (2) M.R. Luo and R.W.G. Hunt, "A chromatic adaptation transform and a 
      colour inconstancy index." COLOR research and application, 1997.

    Arguments:
    (initialization)
      XYZ_test: (tuple) A 3-tuple of the CIEXYZ coordinates for the test 
        illuminant.
      XYZ_reference: (tuple) A 3-tuple of the CIEXYZ coordinates for the reference 
        illuminant.

    (forward)
      im: (torch.Tensor) An NCHW torch tensor, where C presents the CIEXYZ 
        coordinates of the image.'''
    def __init__(self, XYZ_test, XYZ_reference):
        super().__init__()
        M = torch.tensor([[0.8951, 0.2664, -0.1614],
                          [-0.7502, 1.7135, 0.0367],
                          [0.0389, -0.0685, 1.0296]]).transpose(0,1)
        with torch.no_grad():
            M_inv = torch.inverse(M)
        self.register_buffer('M', M)
        self.register_buffer('M_inv', M_inv)
        self.register_buffer('XYZ_test', torch.tensor(XYZ_test, dtype=torch.float32))
        self.register_buffer('XYZ_reference', torch.tensor(XYZ_reference, dtype=torch.float32))

    def forward(self, im):
        batch = im.shape[0]
        h = im.shape[2]
        w = im.shape[3]
        assert im.shape[1] == 3
        im = im.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)
        scaled_im = im.clone()
        scaled_im[:,0] /= im[:,1]
        scaled_im[:,1] /= im[:,1]
        scaled_im[:,2] /= im[:,1]
        lms_im = torch.matmul(scaled_im,self.M)
        lms_w = torch.matmul(self.XYZ_test, self.M)
        lms_wr = torch.matmul(self.XYZ_reference, self.M)

        # LMS responses for sample (R_c, G_c, B_c)
        lms_im[:,0] = lms_wr[0]/lms_w[0]*lms_im[:,0]
        lms_im[:,1] = lms_wr[1]/lms_w[1]*lms_im[:,1]
        p = (lms_w[2]/lms_wr[2])**0.0834
        lms_im[:,2] = (lms_wr[2]/lms_w[2]**p)*torch.abs(lms_im[:,2])**p

        lms_im[:,0] *= im[:,1]
        lms_im[:,1] *= im[:,1]
        lms_im[:,2] *= im[:,1]

        XYZ = torch.matmul(lms_im, self.M_inv)
        XYZ = XYZ.reshape(batch, h, w, 3).permute(0,3,1,2)
        return XYZ

class TemperatureShiftCIEXYZ(torch.nn.Module):
    '''Adapt the color temperature of an image taken under illumination from
    white light at a test color temperature to appear as if it were taken under 
    illumination from white light at a reference color temperature. The 
    chromaticity of the approximately black-body lights at the test and 
    reference color temperatures are calculated using the method of (1). 

    This module operates on CIEXYZ images.

    Arguments:
    
    (initialization)
      test_temp: (float) The temperature of the test (original) illuminant in 
        Kelvin.
      reference_temp: (float) The temperature of the reference (new) illuminant
        in Kelvin.

    (forward)
      im: (torch.Tensor) An NCHW tensor containing the image(s) to be
        transformed. The C channel must contain CIEXYZ coordinates.

    References: 
      (1) B. Kang, O. Moon, C. Hong, H. Lee, B. Cho, and Y.-s. Kim. "Design of 
        advanced color: Temperature control system for HDTV applications." 
        Journal of the Korean Physicial Society. 2002.
        ''' 
    def __init__(self, test_temp, reference_temp):
        super().__init__()
        self.test_temp = test_temp
        self.reference_temp = reference_temp
        xy_test = colour.CCT_to_xy(test_temp, "Kang 2002")
        xy_ref = colour.CCT_to_xy(reference_temp, "Kang 2002")

        X_test = xy_test[0]/xy_test[1]
        Y_test = 1
        Z_test = (1-xy_test[0]-xy_test[1])/xy_test[1]
        XYZ_test = (X_test, Y_test, Z_test)

        X_ref = xy_ref[0]/xy_ref[1]
        Y_ref = 1
        Z_ref = (1-xy_ref[0]-xy_ref[1])/xy_ref[1]
        XYZ_reference = (X_ref, Y_ref, Z_ref)

        self.adaptation = BradfordAdaptation(XYZ_test, XYZ_reference)

    def forward(self, im):
        return self.adaptation(im)

class TemperatureShiftSRGB(TemperatureShiftCIEXYZ):
    '''Adapt the color temperature of an image taken under illumination from 
    white light at a test color temperature to appear as if it were taken under
    illumination from white light at a reference color temperature. The 
    chromaticity of the approximately black-body lights at the test and 
    reference color temperatures are calculated using the method of (1).

    This module operates on sRGB images.

    Arguments:
    
    (initialization)
      test_temp: (float) The temperature of the test (original) illuminant in 
        Kelvin.
      reference_temp: (float) The temperature of the reference (new) illuminant
        in Kelvin.

    (forward)
      im: (torch.Tensor) An NCHW tensor containing the image(s) to be
        transformed. The C channel must contain sRGB coordinates.

    References: 
      (1) B. Kang, O. Moon, C. Hong, H. Lee, B. Cho, and Y.-s. Kim. "Design of 
        advanced color: Temperature control system for HDTV applications." 
        Journal of the Korean Physicial Society. 2002.'''
    def __init__(self, test_temp, reference_temp):
        super().__init__(test_temp, reference_temp)
        self.srgb_to_ciexyz = SRGBToCIEXYZ()
        self.ciexyz_to_srgb = CIEXYZToSRGB()

    def forward(self, im):
        im_xyz = self.srgb_to_ciexyz(im)
        shifted_xyz = super().forward(im_xyz)
        shifted_srgb = self.ciexyz_to_srgb(shifted_xyz)
        return shifted_srgb

class TemperatureShiftTorch(TemperatureShiftSRGB):
    '''
    Same as TemperatureShiftSRGB, except with inputs in range [-1, 1] instead 
    of standard sRGB [0,1].

    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor containing the image(s) to be
        transformed. The C channel must contain [-1, 1]-scaled sRGB coordinates
        (that is, the native scaling of images parsed by torchvision).
    '''
    def forward(self, im):
        im = torchvision_to_unit_interval(im)
        return unit_interval_to_torchvision(super().forward(im))

class ChromaticityShiftSRGB(torch.nn.Module):
    """Shift the color of an input image by adding offsets to u* and v* in the 
    CIELUV color space. The module calculates the CIE L*u*v* coordinates of an 
    input image, which is assumed to be in the sRGB space, and then calculates 
    a color-shifted image as:

    L* = L*
    u* += delta_u_star
    v* += delta_v_star

    The image is then converted back to sRGB before returning.

    Arguments:

    (initialization):
      delta_u_star: (float) An offset for the u* coordinate.
      delta_v_star: (float) An offset for the v* coordinate.

    (forward):
      im: (torch.Tensor) An NCHW image, where C represents the sRGB coordinates
      of an image."""
    def __init__(self, delta_u_star, delta_v_star):
        super().__init__()
        self.srgb_to_ciexyz = SRGBToCIEXYZ()
        self.ciexyz_to_cieluv = CIEXYZToCIELUV()
        self.cieluv_to_ciexyz = CIELUVToCIEXYZ()
        self.ciexyz_to_srgb = CIEXYZToSRGB()
        self.delta_u_star = delta_u_star
        self.delta_v_star = delta_v_star

    def forward(self, im):
        im_luv = self.ciexyz_to_cieluv(self.srgb_to_ciexyz(im))
        im_luv[:,1] += self.delta_u_star
        im_luv[:,2] += self.delta_v_star
        new_im = self.ciexyz_to_srgb(self.cieluv_to_ciexyz(im_luv))
        return new_im

class ChromaticityShiftTorch(ChromaticityShiftSRGB):
    '''
    Same as ChromaticityShiftSRGB, except with inputs in range [-1, 1] instead 
    of standard sRGB [0,1].

    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor containing the image(s) to be
        transformed. The C channel must contain [-1, 1]-scaled sRGB coordinates
        (that is, the native scaling of images parsed by torchvision).
    '''
    def forward(self, im):
        im = torchvision_to_unit_interval(im)
        return unit_interval_to_torchvision(super().forward(im))

def load_to_torch(path):
    im = Image.open(path)
    im = np.array(im, dtype=np.float32)
    im /= 255
    im = torch.tensor(im)
    return im.permute(2,0,1).unsqueeze(0)

def torch_to_pil(im):
    im = im.detach().permute(0,2,3,1).squeeze(0).cpu().numpy()
    im *= 255
    im = np.require(im, dtype=np.uint8)
    im = Image.fromarray(im)
    return im

def save_torch(im, path):
    im = torch_to_pil(im)
    im.save(path)
