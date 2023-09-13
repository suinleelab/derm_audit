#!/usr/bin/env python
from PIL import Image
import torch
import numpy as np

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
