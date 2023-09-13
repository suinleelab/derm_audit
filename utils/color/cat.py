#!/usr/bin/env python
import torch
import colour

from .bradford import BradfordAdaptation
from .cam16 import CAM16CAT
from .ciexyz import SRGBToCIEXYZ, CIEXYZToSRGB
from .ciexyz import torchvision_to_unit_interval, unit_interval_to_torchvision
from .cielab import CIEXYZToCIELAB, CIELABToCIEXYZ
from .cieluv import CIEXYZToCIELUV, CIELUVToCIEXYZ

class CIELABAdaptationTorch(torch.nn.Module):
    def __init__(self, xyz_n):
        super().__init__()
        self.srgb_to_ciexyz = SRGBToCIEXYZ()
        self.ciexyz_to_cielab = CIEXYZToCIELAB(xyz_n)
        self.cielab_to_ciexyz = CIELABToCIEXYZ()
        self.ciexyz_to_srgb = CIEXYZToSRGB()

    def forward(self, im):
        temp = torchvision_to_unit_interval(im)
        temp = self.srgb_to_ciexyz(temp)
        temp = self.ciexyz_to_cielab(temp) # this step includes chromatic adaptation transform
        temp = self.cielab_to_ciexyz(temp) 
        temp = self.ciexyz_to_srgb(temp)
        return unit_interval_to_torchvision(temp)

class CIELUVAdaptationSRGB(torch.nn.Module):
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

class CIELUVAdaptationTorch(CIELUVAdaptationSRGB):
    '''
    Same as ChromaticityShiftSRGB, except with inputs in range [-1, 1] instead 
    of standard sRGB [0,1].

    Arguments:
    (forward)
      im: (torch.Tensor) An NCHW tensor containing the image(s) to be
        transformed. The C channel must contain [-1, 1]-scaled sRGB coordinates
    '''
    def forward(self, im):
        im = torchvision_to_unit_interval(im)
        return unit_interval_to_torchvision(super().forward(im))

class CAT16Torch(CAM16CAT):
    '''
    Chromatic adaption transform from the Color Appearance Model 16 (CAM16),
    for use with inputs in the range [-1, 1]

    Arguments:
    (forward)
      im:
      im: (torch.Tensor) An NCHW tensor containing the image(s) to be
        transformed. The C channel must contain [-1, 1]-scaled sRGB coordinates
    '''
    def __init__(self, xyz_w, xyz_wr, D=1, surround='average', L_A=300):
        super().__init__(xyz_w, xyz_wr, surround=surround, L_A = L_A)
        self.srgb_to_ciexyz = SRGBToCIEXYZ()
        self.ciexyz_to_srgb = CIEXYZToSRGB()

    def forward(self, im):
        temp = torchvision_to_unit_interval(im)
        temp = self.srgb_to_ciexyz(temp)
        temp = super().forward(temp)
        temp = self.ciexyz_to_srgb(temp)
        return unit_interval_to_torchvision(temp)

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
    '''
    def forward(self, im):
        im = torchvision_to_unit_interval(im)
        return unit_interval_to_torchvision(super().forward(im))
