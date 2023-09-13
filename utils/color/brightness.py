#!/usr/bin/env python
import torch

from .ciexyz import SRGBToCIEXYZ, CIEXYZToSRGB
from .ciexyz import torchvision_to_unit_interval, unit_interval_to_torchvision
from .ciexyz import srgb_to_linear_rgb, linear_rgb_to_srgb
from .cieluv import CIEXYZToCIELUV, CIELUVToCIEXYZ
from .jzazbz import CIEXYZToJzazbz, JzazbzToCIEXYZ

class LightnessShiftSRGB(torch.nn.Module):
    """Shift the lightness of an input image by adding offsets to L* in the 
    CIELUV color space. The module calculates the CIE L*u*v* coordinates of an 
    input image, which is assumed to be in the sRGB space, and then calculates 
    a lightness-shifted image as:

    L* += delta_L_star
    u* = u*
    v* = v*

    The image is then converted back to sRGB before returning.

    Arguments:

    (initialization):
      delta_L_star: (float) An offset for the L* coordinate.

    (forward):
      im: (torch.Tensor) An NCHW image, where C represents the sRGB coordinates
      of an image (scaled to the range [0,1])."""
    def __init__(self, delta_L_star):
        super().__init__()
        self.srgb_to_ciexyz = SRGBToCIEXYZ()
        self.ciexyz_to_cieluv = CIEXYZToCIELUV()
        self.cieluv_to_ciexyz = CIELUVToCIEXYZ()
        self.ciexyz_to_srgb = CIEXYZToSRGB()
        self.delta_L_star = delta_L_star

    def forward(self, im):
        im_luv = self.ciexyz_to_cieluv(self.srgb_to_ciexyz(im))
        im_luv[:,0] += self.delta_L_star
        new_im = self.ciexyz_to_srgb(self.cieluv_to_ciexyz(im_luv))
        return new_im

class LightnessShiftTorch(LightnessShiftSRGB):
    """Shift the lightness of an input image by adding offsets to L* in the 
    CIELUV color space. The module calculates the CIE L*u*v* coordinates of an 
    input image, which is assumed to be in the sRGB space, and then calculates 
    a color-shifted image as:

    L* += delta_L_star
    u* = u*
    v* = v*

    The image is then converted back to sRGB before returning.

    Arguments:

    (initialization):
      delta_L_star: (float) An offset for the L* coordinate.

    (forward):
      im: (torch.Tensor) An NCHW image, where C represents the sRGB coordinates
      of an image (scaled to the range [-1,1]."""
    def forward(self, im):
        im = torchvision_to_unit_interval(im)
        return unit_interval_to_torchvision(super().forward(im))

class BrightnessRGBShiftTorch(torch.nn.Module):
    '''Shift exposure of image via multiplication in linear rgb space.
    Arguments:
      exposure_change: The brightness factor "n", where we multiple the linear 
        rgb values by (2**n)
    '''

    def __init__(self, exposure_change):
        super().__init__()
        self.exposure_change = exposure_change

    def forward(self, im):
        #batch = im.shape[0]
        #h = im.shape[2]
        #w = im.shape[3]
        #assert im.shape[1] == 3

        temp = torchvision_to_unit_interval(im)

        temp = temp.permute(0,2,3,1)
        temp = srgb_to_linear_rgb(temp)
        temp = temp*(2**self.exposure_change)
        return unit_interval_to_torchvision(linear_rgb_to_srgb(temp).permute(0,3,1,2))

class JzazbzShiftTorch(torch.nn.Module):
    def __init__(self, delta_jz, delta_az, delta_bz):
        super().__init__()
        self.delta_jz = delta_jz
        self.delta_az = delta_az
        self.delta_bz = delta_bz
        self.srgb_to_ciexyz = SRGBToCIEXYZ()
        self.ciexyz_to_jzazbz = CIEXYZToJzazbz()
        self.jzazbz_to_ciexyz = JzazbzToCIEXYZ()
        self.ciexyz_to_srgb = CIEXYZToSRGB()

    def forward(self, im):
        temp = torchvision_to_unit_interval(im)
        temp = self.srgb_to_ciexyz(temp)
        temp = self.ciexyz_to_jzazbz(temp)

        temp[:,0] += self.delta_jz
        temp[:,1] += self.delta_az
        temp[:,2] += self.delta_bz

        temp = self.jzazbz_to_ciexyz(temp)
        temp = self.ciexyz_to_srgb(temp)
        return unit_interval_to_torchvision(temp)

class JzScaleTorch(torch.nn.Module):
    '''
    Shift exposure of image via multiplication of the Jz channel in Jzazbz space.

    Arguments:
      exposure_change: The brightness factor "n", where we multiple the Jz 
        values by (2**n)
    '''
    def __init__(self, delta_jz):
        super().__init__()
        self.delta_jz = delta_jz
        self.srgb_to_ciexyz = SRGBToCIEXYZ()
        self.ciexyz_to_jzazbz = CIEXYZToJzazbz()
        self.jzazbz_to_ciexyz = JzazbzToCIEXYZ()
        self.ciexyz_to_srgb = CIEXYZToSRGB()

    def forward(self, im):
        temp = torchvision_to_unit_interval(im)
        temp = self.srgb_to_ciexyz(temp)
        temp = self.ciexyz_to_jzazbz(temp)

        temp[:,0] *= 2**self.delta_jz

        temp = self.jzazbz_to_ciexyz(temp)
        temp = self.ciexyz_to_srgb(temp)
        return unit_interval_to_torchvision(temp)
