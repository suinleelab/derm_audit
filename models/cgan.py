#!/usr/bin/env python
"""
PyTorch implementation of conditional generative adversarial networks.
"""
import torch
from torch import nn

class CategoricalConditionalBatchNorm(nn.Module):
    __doc__ = r"""Apply categorical conditional batch normalization.

    This module applies standard 2-dimensional batch normalization, then 
    linearly transforms each item in the batch by a scale and offset that depend
    upon that item's class label. For an input of size :math:`(N, C, H, W)`, the
    output will also have save :math:`(N, C, H, W)` and is given by:

    .. math::

        \text{out}(N_i) = x(N_i) * \gamma(y(N_i)) + \beta(y(N_i))

    where :math:`\gamma` and :math:`\beta` map the categorical y-values to a 
    real-valued scale and offset. The parameters :math:`\gamma` and 
    :math:`\beta` are learnable.

    Args:
        num_classes (int): The number of categorial y-values. The y values are 
            expected to be scaled between 0 and `num_classes`.
        channels (int): The number of input channels. The output will contain 
            the same number of channels."""
    def __init__(self, num_classes, channels):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.batchnorm = nn.BatchNorm2d(self.channels, affine=False)

        # lookup tables. maps the y label (an int in range(0, num_classes))
        # to the vector of gammas or betas
        self.embed_gamma = nn.Embedding(self.num_classes, self.channels)
        self.embed_beta = nn.Embedding(self.num_classes, self.channels)

        # initialize 
        #nn.init.normal_(self.embed_gamma.weight, mean=1, std=0.02)
        nn.init.constant_(self.embed_gamma.weight, 1)
        nn.init.zeros_(self.embed_beta.weight)

    def forward(self, x, y):
        temp = self.batchnorm(x)
        gamma = self.embed_gamma(y)
        beta = self.embed_beta(y)
        return gamma.view(-1, self.channels, 1, 1)*temp + beta.view(-1, self.channels, 1, 1)

class DownsamplingBlock(nn.Module):
    __doc__ = r"""Downsampling block for use in the generator network.
    
    This module downsamples incoming tensors via a 3 by 3 convolutional filter 
    with stride 2, then applies categorial conditional batch normalization and a
    regularized linear unit activation."""
    def __init__(self, num_classes=10, in_channels=64):
        super().__init__()
        self.num_classes = num_classes 
        out_channels = in_channels*2
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                          stride=2, padding=1)
        self.layer2 = CategoricalConditionalBatchNorm(self.num_classes, out_channels)
        self.layer3 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        return self.layer3(self.layer2(self.layer1(x),y))

class UpsamplingBlock(nn.Module):
    __doc__ = r"""Downsampling block for use in the generator network.
    
    This module upsamples incoming tensors via a 3 by 3 convolutional filter 
    with stride 1/2 (that is, a tranposed convolution), then applies categorial 
    conditional batch normalization and a regularized linear unit activation."""
    def __init__(self, num_classes=10, in_channels=64):
        super().__init__()
        self.num_classes = num_classes
        out_channels = int(in_channels/2)

        self.layer1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                   stride=2, padding=1, output_padding=1)

        self.layer2 = CategoricalConditionalBatchNorm(num_classes, out_channels)
        self.layer3 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        return self.layer3(self.layer2(self.layer1(x),y))

class ResBlock(nn.Module):
    __doc__ = r"""Residual block for use in the generator network.
    
    This module applies two 3 by 3 convolutions, separated by a regularized 
    linear unit activation function. Each convolution is also followed by 
    categorial conditional batch normalization."""
    def __init__(self, num_classes=10, channels=256):
        super().__init__()
        self.num_classes = num_classes
        self.padding_layer = nn.ReflectionPad2d(1)
        self.layer1 = nn.Conv2d(channels, channels, kernel_size=3)
                
        self.layer2 = CategoricalConditionalBatchNorm(self.num_classes, channels)
        self.layer3 = nn.ReLU(inplace=True)
        # padding layer goes in between 3 and 4; defined in forward method
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=3)
        self.layer5 = CategoricalConditionalBatchNorm(self.num_classes, channels)

    def forward(self, x, y):
        temp = self.padding_layer(x)
        temp = self.layer1(temp)
        temp = self.layer2(temp, y)
        temp = self.layer3(temp)
        temp = self.padding_layer(temp)
        temp = self.layer4(temp)
        temp = self.layer5(temp, y)
        return temp + x


class Generator(nn.Module):
    __doc__ = r"""Generator network, based on fusion of the CycleGAN generator
    architecture from ref (1) and the explanation by progressive exaggeration 
    generator architecture from ref (2). 

    This module transforms a batch of images :math:`x`, dependent on a vector of
    integer-valued category labels :math:`y`.

    Args:
        in_channels (int, optional): The number of channels in the input image.
        n_resblocks (int, optional): The number of residual blocks.
        num_classes (int, optional): THe number of classes to generate.
    
    References:
    (1) Zhu, Park, Isola, and Efros. "Unpaired image-to-image translation using 
        cycle-consistent adversarial networks" ICCV 2017, arXiv:1703.10593 
    (2) Singla, Pollack, Chen, and Batmanghelich. "Explanation by progessive 
        exaggeration" ICLR 2020."""
    def __init__(self, in_channels=3, n_resblocks=9, num_classes=10, im_size=224):
        super().__init__()
        self.n_resblocks = n_resblocks
        self.num_classes = num_classes
        self.im_size = im_size

        # initial convolution
        self.layer1 = nn.ReflectionPad2d(3)
        self.layer2 = nn.Conv2d(in_channels, 64, kernel_size=7)
        self.layer3 = CategoricalConditionalBatchNorm(self.num_classes, 64)
        self.layer4 = nn.ReLU(inplace=True)

        # downsampling 1
        self.layer5 = DownsamplingBlock(num_classes=self.num_classes, in_channels=64)

        # downsampling 2
        self.layer6 = DownsamplingBlock(num_classes=self.num_classes, in_channels=128)

        # residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(n_resblocks):
            self.resblocks.append(ResBlock(num_classes=self.num_classes, channels=256))

        # upsampling 1
        self.layer7 = UpsamplingBlock(num_classes=self.num_classes, in_channels=256)

        # upsampling 2
        self.layer8 = UpsamplingBlock(num_classes=self.num_classes, in_channels=128)

        # final conv
        self.layer9 = nn.ReflectionPad2d(3)
        self.layer10 = nn.Conv2d(64, in_channels, kernel_size=7)
        self.layer11 = nn.Tanh()

        ## layer initializations ##
        # conv
        for layer in [self.layer2, self.layer10, self.layer5.layer1, 
                self.layer6.layer1, self.layer7.layer1, self.layer8.layer1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # resblocks
        for resblock in self.resblocks:
            nn.init.xavier_uniform_(resblock.layer1.weight)
            nn.init.zeros_(resblock.layer1.bias)
            nn.init.xavier_uniform_(resblock.layer4.weight)
            nn.init.zeros_(resblock.layer4.bias)

    def forward(self, x, y):
        temp = self.layer1(x)
        temp = self.layer2(temp)
        temp = self.layer3(temp, y)
        temp = self.layer4(temp)
        temp = self.layer5(temp, y)
        temp = self.layer6(temp, y)
        for i in range(self.n_resblocks):
            temp = self.resblocks[i](temp, y)
        embedding = temp
        temp = self.layer7(temp, y)
        temp = self.layer8(temp, y)
        temp = self.layer9(temp)
        temp = self.layer10(temp)
        temp = self.layer11(temp)
        return temp[:,:,:self.im_size,:self.im_size], embedding

class Discriminator(nn.Module):
    __doc__ = r"""Discriminator network, based on fusion of the CycleGAN 
    discriminator architecture from ref (1) and the explanation by progressive 
    exaggeration discriminator architecture from ref (2). 

    This module predicts whether each item in a batch of images :math:`x` is 
    real or artificially generated, dependent on a vector of integer-valued 
    category labels :math:`y`.

    Args:
        in_channels (int, optional): The number of channels in the input image.
        num_classes (int, optional): THe number of classes to generate.
    
    References:
    (1) Zhu, Park, Isola, and Efros. "Unpaired image-to-image translation using 
        cycle-consistent adversarial networks" ICCV 2017, arXiv:1703.10593 
    (2) Singla, Pollack, Chen, and Batmanghelich. "Explanation by progessive 
        exaggeration" ICLR 2020."""
    def __init__(self, in_channels=3, num_classes=10):
        self.in_channels = in_channels
        self.num_classes = 10
        super().__init__()
        model = [nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.utils.parametrizations.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.utils.parametrizations.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.utils.parametrizations.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]

        self.model = nn.Sequential(*model)

        # custom initialization for convolutions
        for i in [0,2,4,6]:
            nn.init.xavier_uniform_(self.model[i].weight)
            nn.init.zeros_(self.model[i].bias)


        # linear layer
        self.linear = nn.utils.parametrizations.spectral_norm(
                nn.Linear(512, 1, bias=True)
                )
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def turn_off_sn(self):
        r"""Stop spectral normalization of the layer weights.
        
        When spectral normalization is on, layer weights are updated each time
        they are accessed, including on forward calls. This causes backward 
        passes to fail when the backward pass depends upon multiple forward 
        calls to the network. Multiple forward calls per backward pass are a 
        typical scenario, since the disciminator loss may depend upon 
        predictions from both a batch of real images and a batch of artificially
        generated images."""
        for i in [0,2,4,6]:
            self.model[i].eval()
        self.linear.eval()

    def turn_on_sn(self):
        r"""Turn on spectral normalization of the layer weights. See 
        `turn_off_sn` for potential problem scenarios."""
        for i in [0,2,4,6]:
            self.model[i].train()
        self.linear.train()

    def forward(self, x, y):
        temp = self.model(x)

        # sum over spatial dimensions (not batch or channel)
        # creates tensor of shape (batch_size, channels)
        temp = torch.sum(temp, [2,3]) 
        return self.linear(temp).squeeze(1)

class ClassConditionalDiscriminator(Discriminator):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(self, in_channels=in_channels, num_classes=num_classes)
        # projection layer
        self.projection = nn.utils.parametrizations.spectral_norm(
                nn.Embedding(2, 512)
                )
        nn.init.xavier_uniform_(self.projection.weight)

    def get_binary(self, y):
        '''map an integer y to a one-hot mask that respects ordering of the 
        classes. The first element of the vector is equal to the label. For 
        instance, if y=3, then the mask is [3,1,1,1,0,0,0,0,0,0]'''
        binary_mask = torch.zeros((y.shape[0], self.num_classes), device=next(self.parameters()).device, dtype=torch.long)
        binary_mask[:,0] = y
        for i in range(y.shape[0]):
            binary_mask[i, 1:y[i]+1] = 1
        return binary_mask

    def forward(self, x, y):
        temp = self.model(x)

        # sum over spatial dimensions (not batch or channel)
        # creates tensor of shape (batch_size, channels)
        temp = torch.sum(temp, [2,3]) 
        mask = self.get_binary(y)

        # following singla et al
        proj = self.projection(mask[:,1])*temp
        self.projection.eval() # stop updating spectral radius
        for i in range(2,mask.shape[1]):
            proj += self.projection(mask[:,i])*temp
        self.projection.train()
        return torch.sum(proj,1) + self.linear(temp).squeeze(1)

class GResBlockEncoder(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels):
        super().__init__()
        self.layer1 = CategoricalConditionalBatchNorm(num_classes, in_channels)
        self.layer2 = nn.ReLU(inplace=True)
        self.layer3 = nn.AvgPool2d(2) # may have different padding behavior than original implementation
        self.layer4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', stride=1)
        self.layer5 = CategoricalConditionalBatchNorm(num_classes, out_channels)
        self.layer6 = nn.ReLU(inplace=True)
        self.layer7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', stride=1)

        # residual connection
        self.layer1b = nn.AvgPool2d(2)
        self.layer2b = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', stride=1)

    def forward(self, x, y):
        temp = self.layer1(x, y)
        temp = self.layer2(temp)
        temp = self.layer3(temp)
        temp = self.layer4(temp)
        temp = self.layer5(temp, y)
        temp = self.layer6(temp)
        temp = self.layer7(temp)

        # residual connection
        temp2 = self.layer1b(x)
        temp2 = self.layer2b(temp2)
        return temp + temp2

class GResBlockDecoder(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels):
        super().__init__()
        self.layer1 = CategoricalConditionalBatchNorm(num_classes, in_channels)
        self.layer2 = nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Upsample(scale_factor=2)
        self.layer4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', stride=1)
        self.layer5 = CategoricalConditionalBatchNorm(num_classes, out_channels)
        self.layer6 = nn.ReLU(inplace=True)
        self.layer7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', stride=1)

        self.layer1b = nn.Upsample(scale_factor=2)
        self.layer2b = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', stride=1)

    def forward(self, x, y):
        temp = self.layer1(x, y)
        temp = self.layer2(temp)
        temp = self.layer3(temp)
        temp = self.layer4(temp)
        temp = self.layer5(temp, y)
        temp = self.layer6(temp)
        temp = self.layer7(temp)

        temp2 = self.layer1b(x)
        temp2 = self.layer2b(temp2)
        return temp + temp2

class DResBlock(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, downsample=True, first_resblock=False):
        super().__init__()
        self.downsample = downsample
        self.first_resblock = first_resblock

        self.layer1 = nn.ReLU(inplace=True)
        self.layer2 = nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', stride=1)
                )
        self.layer3 = nn.ReLU(inplace=True)
        self.layer4 = nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', stride=1)
                )
        if downsample:
            self.layer5 = nn.AvgPool2d(2)

            self.layer1b = nn.utils.parametrizations.spectral_norm(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', stride=1)
                    )
            self.layer2b = nn.AvgPool2d(2)

    def forward(self, x):
        if self.first_resblock:
            temp = x
        else:
            temp = self.layer1(x)
        temp = self.layer2(temp)
        temp = self.layer3(temp)
        temp = self.layer4(temp)
        if self.downsample:
            temp = self.layer5(temp)

            if self.first_resblock: # average pool then conv
                temp2 = self.layer2b(x)
                temp2 = self.layer1b(temp2)
            else: # conv then average pool
                temp2 = self.layer1b(x)
                temp2 = self.layer2b(temp2)
            return temp + temp2
        else:
            return temp + x

class ResNetGenerator(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, im_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.im_size = im_size

        self.layer1 = CategoricalConditionalBatchNorm(self.num_classes, in_channels)
        self.layer2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Conv2d(in_channels, 64, kernel_size=3, padding='same', stride=1)
        self.layer4 = GResBlockEncoder(num_classes, 64, 128)
        self.layer5 = GResBlockEncoder(num_classes, 128, 256)
        self.layer6 = GResBlockEncoder(num_classes, 256, 512)
        self.layer7 = GResBlockEncoder(num_classes, 512, 1024)
        self.layer8 = GResBlockEncoder(num_classes, 1024, 1024)

        self.layer9 = GResBlockDecoder(num_classes, 1024, 1024)
        self.layer10 = GResBlockDecoder(num_classes, 1024, 512)
        self.layer11 = GResBlockDecoder(num_classes, 512, 256)
        self.layer12 = GResBlockDecoder(num_classes, 256, 128)
        self.layer13 = GResBlockDecoder(num_classes, 128, 64)
        self.layer14 = CategoricalConditionalBatchNorm(self.num_classes, 64)
        self.layer15 = nn.ReLU(inplace=True)
        self.layer16 = nn.Conv2d(64, in_channels, kernel_size=3, padding='same', stride=1)
        self.layer17 = nn.Tanh()

    def forward(self, x, y):
        temp = self.layer1(x, y)
        temp = self.layer2(temp)
        temp = self.layer3(temp)
        temp = self.layer4(temp, y)
        temp = self.layer5(temp, y)
        temp = self.layer6(temp, y)
        temp = self.layer7(temp, y)
        temp = self.layer8(temp, y)
        embedding = temp

        temp = self.layer9(temp, y)
        temp = self.layer10(temp, y)
        temp = self.layer11(temp, y)
        temp = self.layer12(temp, y)
        temp = self.layer13(temp, y)
        temp = self.layer14(temp, y)
        temp = self.layer14(temp, y)
        temp = self.layer15(temp)
        temp = self.layer16(temp)
        return self.layer17(temp), embedding

class ResNetDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, im_size=224):
        prepool_size = int(im_size/2**5)
        if not prepool_size == im_size/2**5:
            raise ValueError("downsampling operations reduce input size by factor of 2**5, but {:d}/2**5 is not an integer!".format(im_size))
        super().__init__()
        self.num_classes = num_classes

        self.layer1 = DResBlock(num_classes, in_channels, 64, downsample=True, first_resblock=True)
        self.layer2 = DResBlock(num_classes, 64, 128, downsample=True)
        self.layer3 = DResBlock(num_classes, 128, 256, downsample=True)
        self.layer4 = DResBlock(num_classes, 256, 512, downsample=True)
        self.layer5 = DResBlock(num_classes, 512, 1024, downsample=True)
        self.layer6 = DResBlock(num_classes, 1024, 1024, downsample=False)
        self.layer7 = nn.ReLU(inplace=True)
        self.layer8 = nn.AvgPool2d(prepool_size, divisor_override=1)
        self.layer9 = nn.utils.parametrizations.spectral_norm(
                nn.Linear(1024, 1, bias=True)
                )

    def turn_off_sn(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            layer.layer2.eval()
            layer.layer4.eval()
            layer.layer1b.eval()
        self.layer6.layer2.eval()
        self.layer6.layer4.eval()
        self.layer9.eval()

    def turn_on_sn(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            layer.layer2.train()
            layer.layer4.train()
            layer.layer1b.train()
        self.layer6.layer2.train()
        self.layer6.layer4.train()
        self.layer9.train()

    def forward(self, x, y):
        temp = self.layer1(x)
        temp = self.layer2(temp)
        temp = self.layer3(temp)
        temp = self.layer4(temp)
        temp = self.layer5(temp)
        temp = self.layer6(temp)
        temp = self.layer7(temp)
        temp = self.layer8(temp)
        temp = self.layer9(temp.squeeze(3).squeeze(2))
        return temp
