#!/usr/bin/env python
import numpy as np
import torch

import models.caffe_pb2 as caffe_pb2
import models.c2p as c2p

class ModelDermClassifier(torch.nn.Module):
    '''Wrapper for ModelDerm 2018 classifier; for use with (-1,1)-scaled 
    images'''
    def __init__(self):
        super().__init__()
        self.image_size = 224
        self.positive_index = 140
        self.mean_path = "/projects/leelab3/derm/modelderm_2018/model/"\
                         "asanplus/mean224x224.binaryproto"
        self.equalize = Equalize()
        # Load mean image
        blob = caffe_pb2.BlobProto()
        with open(self.mean_path, 'rb') as f:
            blob.ParseFromString(f.read())
        mean = np.array(blob.data)
        mean = mean.reshape([blob.channels, blob.height, blob.width])
        mean = torch.tensor(mean, dtype=torch.float32)
        # Register mean image to ensure transfer on calls to `self.to(device)`
        self.register_buffer('mean', mean)

        # Load model
        self.model = c2p.ConvertedCaffeModel(
                './pretrained_classifiers/deploy.prototxt', 
                './pretrained_classifiers/70616.caffemodel')
        self.eval()

    def forward(self, x):
        # Rescale to (0,255 range)
        temp = 127.5*x+127.5 # 255*(x/2+0.5)
        # RGB to BGR
        temp = temp[:,[2,1,0]]

        # Differentiable variant of histogram equalization
        temp = self.equalize(temp)

        # Subtract mean image
        temp -= self.mean
        # Return result 
        return self.model(temp)

class SoftLessThan(torch.nn.Module):
    """A continuous variant of the "less than" operator.
    
    In place of the hard discontinuity of the usual "less than" operator 
    `x < y` at the point where `x = y`, this module linearly interpolates
    between 1 and 0 using a line of the specified slope.

    In particular, this module implements the function:
        
        f(x, y) := 1              ; x < y - 0.5/m
                   m*(y-x+0.5/m)  ; y - 0.5/m < x < y + 0.5/m
                   0              ; x > y + 0.5/m

    Args:
        slope (float): The slope of the linear interpolation.
                   """
    def __init__(self, slope=1):
        super().__init__()
        self.register_buffer("slope", torch.tensor(slope))
        self.relu = torch.nn.ReLU()

    def forward(self, x, y):
        return torch.clamp(self.slope*self.relu(0.5/self.slope + y - x), min=0, max=1)

class Equalize(torch.nn.Module):
    """Perform histogram equalization on an image, in a differentiable manner.

    Typical histogram equalization modifies an integer-valued image by 
    "flattening" the probability mass function of its pixels. The typical 
    version of histogram equalization achieves this by viewing the probability 
    mass function as an approximation of a probability density function `f`, 
    which is then integrated to obtain the cumulative density function `F`.
    Each pixel `p` is mapped to its percentile `F(p)` and then rescaled to 
    cover the image's native range (e.g., 0-255 rather than the CDF's range of
    0-1).

    To achieve a differentiable analogue of histogram equalization that is 
    compatible with floating point-valued images, this module estimates the
    cumulative distribution function `F` as a series of line segments with
    endpoints evenly distributed along a grid. The value of the CDF estimate
    `F_hat` at each endpoint is calculated by kernel density estimation with
    a triangular kernel, and values between the endpoints are linearly 
    interpolated.

    ..note: Each channel of the image is equalized independently.

    Args:
        im_size (int): The height and width of the (square) input images.
        segments (int): The number of line segments to use for estimation of
            the CDF.
        eps (float): A small value used to prevent division by zero errors.
        max_ (float): The maximum value of the image's range (default 255). The
            minimum value is assumed to be zero.
        slope (float): The slope of the triangular kernel. Should be 
            positive."""
    def __init__(self, im_size=224, segments=20, eps=1e-5, max_=255, slope=0.1):
        super().__init__()
        if not slope > 0:
            raise ValueError("Slope must be > 0.")
        self.im_size = im_size
        self.segments = segments
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("max_", torch.tensor(max_))
        grid = torch.linspace(0, self.max_, segments+1).unsqueeze(1)
        self.register_buffer("grid", grid)
        self.less = SoftLessThan(slope=slope)

    def scale_channel(self, channel):
        """Equalize a single channel of an image."""
        shape = channel.shape
        # Scale to image's full range
        channel = self.max_*(channel-channel.min())/(channel.max()-channel.min()+self.eps)
        channel = channel.ravel().unsqueeze(0)
        # Calculate value of CDF at each point in self.grid
        F_hat = self.less(channel, self.grid).sum(axis=1)/self.im_size**2
        # To estimate CDF at other points, linearly interpolate between grid points
        for i in range(self.segments):
            mask = torch.logical_and(channel>self.grid[i], channel<self.grid[i+1])
            m = (F_hat[i+1] - F_hat[i])/(self.grid[i+1]-self.grid[i])
            b = F_hat[i] - m*self.grid[i]
            channel[mask] = m*channel[mask]+b 
        channel = channel.view(shape)
        return self.max_*(channel-channel.min())/(channel.max()-channel.min()+self.eps)

    def equalize_single(self, img):
        return torch.stack([self.scale_channel(img[c]) for c in range(img.size(0))])

    def forward(self, img):
        """
        Args:
            img (torch.FloatTensor): A torch tensor of shape (N,C,H,W).
        Returns:
            (torch.FloatTensor): The image with each channel equalized 
            independently."""
        return torch.stack([self.equalize_single(x) for x in img])
