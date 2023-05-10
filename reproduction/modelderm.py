#!/usr/bin/env python
"""
This code is derived, with modification, from the MIT-licensed code written by
Seung Seog Han and hosted publicly at:

https://figshare.com/articles/code/Caffemodel_files_and_Python_Examples/5406223

In particular, this code adapts methods found in the file "test.py" from the 
above repository.


This file is released under the MIT License:

MIT License

    Copyright (c) 2017 Seung Seog Han
    Copyright (c) 2023 Alex DeGrave (modifications)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""
import caffe
from caffe.proto import caffe_pb2
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms

def equalize(img):
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    return img

def pytorch_load(path, mean_img, image_size=224):
    img = Image.open(path)

    # resize and crop
    img = torchvision.transforms.functional.resize(
            img, 
            size=image_size, 
            interpolation=Image.BILINEAR)
    img = torchvision.transforms.functional.center_crop(img, image_size)
    img = np.array(img, dtype=np.uint8)

    # equalize
    img = equalize(img)
    img = np.require(img, dtype=np.float32)

    # channels first
    img = np.transpose(img, (2,0,1))
    # permute rgb to bgr
    img = img[[2,1,0],:,:]
    assert img.shape == (3,224,224)

    return img - mean_img

def native_load(path, mean_img, image_size=224):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = equalize(img)
    img = cv2.resize(img, 
                     (image_size, image_size), 
                     interpolation=cv2.INTER_CUBIC)
    img = np.require(img, dtype=np.float64)
    img = np.transpose(img, (2,0,1))
    return img - mean_img

class CaffeModelDermClassifier(object):
    def __init__(self, 
                 proto_path = "./pretrained_classifiers/deploy.prototxt",
                 weight_path = "./pretrained_classifiers/70616.caffemodel",
                 mean_path = "./pretrained_classifiers/mean224x224.binaryproto"):
        self.mel_index = 140
        self.proto_path = proto_path
        self.weight_path = weight_path
        self.mean_path = mean_path
        self.model = caffe.Net(self.proto_path,
                               self.weight_path,
                               caffe.TEST)
        self.mean_image = self.load_mean_image(self.mean_path)

    def load_mean_image(self, path):
        blob = caffe_pb2.BlobProto()
        with open(path, 'rb') as f:
            blob.ParseFromString(f.read())
        arr = np.array(blob.data)
        return arr.reshape((blob.channels, blob.height, blob.width))

    def run(self, image):
        self.model.blobs['data'].data[...] = image
        return self.model.forward()['prob'][0]

    def run_from_filepath(self, path):
        image = native_load(path, mean_img)
        return self.run(image)
