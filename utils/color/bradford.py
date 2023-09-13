#!/usr/bin/env python
import torch

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
