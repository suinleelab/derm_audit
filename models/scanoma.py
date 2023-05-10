#!/usr/bin/env python
import onnx
import torch

import models.o2p as o2p

class ScanomaClassifier(torch.nn.Module):
    '''Wrapper for Scanoma classifier; for use with (-1,1)-scaled images.'''
    def __init__(self):
        super().__init__()
        self.image_size = 224
        self.positive_index = 1
        self.onnx_path = 'pretrained_classifiers/scanoma.onnx'
        onnx_model = onnx.load(self.onnx_path)
        self.model = o2p.ConvertedModel(onnx_model)
        self.eval()

    def forward(self, x):
        '''Expects an image scaled to (-1,1) range. Scanoma's native input 
        range is also (-1,1), and its output requires no further processing.'''
        return self.model(x)
