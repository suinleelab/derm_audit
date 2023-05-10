#!/usr/bin/env python
import os

import pandas
import torch
import torchvision

class CachedCSVDataset(torch.utils.data.Dataset):
    """Base class for image datasets based on CSV files. 
    
    ..note: self.datapath and self.csvname must be set by a child class before
        calling super().__init__"""
    def __init__(self, transform=None):
        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader
        self.df = pandas.read_csv(os.path.join(self.datapath, self.csvname))
        self.cache = {}
