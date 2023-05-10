#!/usr/bin/env python
#
# Wrapper for ensemble model, similar to winning model of the 2020 SIIM-ISIC 
# Melanoma Classification Kaggle challenge. This model is based on the training 
# scheme described in https://arxiv.org/pdf/2010.05351.pdf and the software 
# distributed by Qishen Ha at 
# https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution. 
#
# This ensemble model comprises 3 individual models trained at a lower 
# resolution than the original winning models. Code in this file is partially 
# based on the code distributed at the above GitHub link, and is redistributed 
# with permission under the following license:
#
#
# MIT License
# 
# Copyright (c) 2020 haqishen
# Copyright (c) 2023 Alex DeGrave (modifications)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import random

import geffnet
import numpy as np
import torch
import torchvision
from torchvision import transforms

# Heavily based on Qishen Ha et al. implementation
class EfficientNet(torch.nn.Module):
    def __init__(self, model_path, enet_type='tf_efficientnet_b7_ns', out_dim=9):
        super().__init__()

        # Set up efficientNet
        self.model_path = model_path
        self.enet = geffnet.create_model(enet_type, pretrained=False)
        self.dropout = torch.nn.Dropout(0.5)

        in_channels = self.enet.classifier.in_features
        self.myfc = torch.nn.Linear(in_channels, out_dim)

        self.enet.classifier = torch.nn.Identity()
        self.softmax = torch.nn.Softmax(dim=1)

        # Load parameters
        # First try loading single GPU model file
        state_dict = torch.load(model_path, map_location='cpu')
        try:
            self.load_state_dict(state_dict, strict=True)
        # Try multiple GPU model file
        except:
            state_dict = {k.replace('module.',''): state_dict[k] for k in state_dict.keys()}
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        logits = self.myfc(self.dropout(self.enet(x).squeeze(-1).squeeze(-1)))
        return self.softmax(logits)

class CalibrationLayer(torch.nn.Module):
    """Calibrate model by mapping model output to the range [0,1] via an 
    empirical cumulative density function. When the layer is called, the  
    output for inputs intermediate the reference inputs are calculated via
    linear interpolation.

    Example:

        >>> ref = torch.tensor([0,0.1,0.2,0.8,0.9,1.0])
        >>> calibration_layer = CalibrationLayer(ref, subsample=None)
        >>> c(ref.unsqueeze(1))
        tensor([[0.0000],
                [0.2000],
                [0.4000],
                [0.6000],
                [0.8000],
                [1.0000]])
        
        >>> c(torch.tensor([0.05,0.15,0.5,0.82]).unsqueeze(1))
        tensor([[0.1000],
                [0.3000],
                [0.5000],
                [0.6400]])
 
    __init__ method:
      Args:
        reference_inputs: (numpy.ndarray or torch.Tensor) The array of model
          outputs on which the base the empirical CDF.
        subsample: (Int or None) Smooth the emprical CDF by subsampling the array
          of reference inputs. After sorting the reference inputs, the reference 
          inputs are divided into ceil(len(reference_inputs)/subsample) blocks,
          and the average of each block becomes a new reference input. Subsampling
          is useful to mitigate extreme gradients that arise when values in the
          array of reference inputs are close together.

    forward method:
      Args:
        x: (torch.Tensor of shape (batch, 1)). The tensor of values to pass
          through the cumulative density function.
      Returns:
        out: (torch.Tensor of shape (batch, 1)). The corresponding CDF 
          estimates.
    """
    def __init__(self, reference_inputs, subsample=100):
        super().__init__()
        reference_inputs, _ = torch.tensor(reference_inputs.flatten()).sort()
        if subsample:
            reference_inputs = [reference_inputs[i*subsample:(i+1)*subsample].mean() for i in range(int(np.ceil(reference_inputs.shape[0]/subsample)))]
            #reference_inputs = reference_inputs[:(reference_inputs.shape[0]//subsample)*subsample].reshape(-1,subsample).mean(axis=1)
            reference_inputs = torch.tensor(reference_inputs)
        reference_outputs = torch.arange(0, len(reference_inputs), dtype=torch.float32)/(len(reference_inputs)-1)
        self.register_buffer('reference_inputs', reference_inputs)
        self.register_buffer('reference_outputs', reference_outputs)

    def forward(self, x):
        # x is shape (batch_size, 1)
        if len(x.shape) != 2:
            print(x)
            raise ValueError("The dimension of the input must be equal to 2")
        gt = (self.reference_inputs>x)
        out = torch.zeros(x.shape, device=x.device)
        for i in range(x.shape[0]):
            # x[i] is greater than all reference inputs
            if x[i] >= self.reference_inputs[-1]:
                out[i] = self.reference_outputs[-1]
            # x[i] is less than all reference inputs
            elif x[i] <= self.reference_inputs[0]:
                out[i] = self.reference_outputs[0]
            else:
                # find index of first reference value greater than x[i]
                idx = torch.argmax(gt[i].to(torch.int16))
                m = (self.reference_outputs[idx] - self.reference_outputs[idx-1])\
                        /(self.reference_inputs[idx] - self.reference_inputs[idx-1])
                out[i] = self.reference_outputs[idx-1] + m*(x[i]-self.reference_inputs[idx-1])
        return out


class SIIMISICClassifier(torch.nn.Module):
    '''Wrapper for model trained is same manner as winning model from the SIIM 
    & ISIC Melanoma Classification Kaggle Challenge. 

    For use with (-1,1)-scaled images.'''
    def __init__(self):
        super().__init__()
        self.image_size = 224
        self.positive_index = 1 # the index of the final output that corresponds to melanoma
        self.mel_idx = 6 # index of the efficientnet output (before ranking) that corresponds to melanoma
        model_paths = ['pretrained_classifiers/9c_b7ns_1e_224_ext_15ep_best_fold0.pth',
                       'pretrained_classifiers/9c_b6ns_1e_224_ext_15ep_best_fold1.pth',
                       'pretrained_classifiers/9c_b5ns_1e_224_ext_15ep_best_fold2.pth']
        model_types = ['tf_efficientnet_b7_ns',
                       'tf_efficientnet_b6_ns',
                       'tf_efficientnet_b5_ns']

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # load model
        self.models = torch.nn.ModuleList([EfficientNet(model_path, enet_type=enet_type) 
                                           for model_path, enet_type in zip(model_paths, model_types)])
        self.augment_list = [lambda img: img,
                             lambda img: img.flip(2),
                             lambda img: img.flip(3),
                             lambda img: img.flip(3).flip(2,3),
                             lambda img: img.transpose(2,3),
                             lambda img: img.transpose(2,3).flip(2),
                             lambda img: img.transpose(2,3).flip(3),
                             lambda img: img.transpose(2,3).flip(3).flip(2,3)]
        self.augment = False
        self.eval()

    def enable_augment(self):
        self.augment = True

    def disable_augment(self):
        self.augment = False

    def forward(self, x):
        '''Expects an image scaled to (-1,1) range. Rescale to the classifer's
        native range, then pass through the classifier and return the softmax of
        the classifier's output'''
        # rescale to (0,1) range
        rescaled = x*0.5+0.5
        # rescale to classifier's native range
        rescaled = self.normalize(rescaled)
        # call the classifier
        if self.augment:
            probs = torch.zeros((x.shape[0], len(self.models), len(self.augment_list))).to(torch.float32).to(x.device)
            for i_augment, augment_fn in enumerate(self.augment_list):
                augmented_img = augment_fn(rescaled)
                for i_model, model in enumerate(self.models):
                    raw = self.models[i_model](augmented_img)[:,self.mel_idx]
                    probs[:,i_model,i_augment] = raw
            probs = probs.mean(2).mean(1)
            # probability of NOT melanoma, probability of melanoma
            return torch.stack((1-probs, probs), dim=1)
        else:
            probs = torch.zeros((x.shape[0], len(self.models))).to(torch.float32).to(x.device)
            for i_model, model in enumerate(self.models):
                augment_fn = random.choice(self.augment_list)
                raw = self.models[i_model](augment_fn(rescaled))[:,self.mel_idx]
                probs[:,i_model] = raw
            # probability of NOT melanoma, probability of melanoma
            probs = probs.mean(dim=1)
            return torch.stack((1-probs, probs), dim=1)
