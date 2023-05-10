#!/usr/bin/env python
import os

from .base import CachedCSVDataset

class ISICDataset(CachedCSVDataset):
    """PyTorch dataset of ISIC 2019 images.

    The dataset caches images after the first load. If loading data using 
    multiple processes via a torch.utils.data.DataLoader instance, make sure 
    the worker processes are persistent (that is, use `persistent_workers=True`
    when initializing the DataLoader).
    
    Args:
        transform (nn.Module subclass): A torch module that transforms the 
            input. Typically, these transforms will be from 
            torchvision.transforms.
        mimics_only (bool): Only include melanomas and melanoma mimickers."""
    def __init__(self, transform=None, mimics_only=True):
        self.datapath = "data/isic/"
        self.csvname = "ISIC_2019_Training_GroundTruth.csv"
        self.dirname = "ISIC_2019_Training_Input"
        self.mimics_only = mimics_only

        super().__init__(transform=transform)
        if self.mimics_only:
            self.df = self.df.query('MEL==1 | NV==1 | BKL==1 | DF==1')

    def _get_label(self, idx):
        label = self.df.MEL.iloc[idx]
        return label

    def __getitem__(self, idx):
        img_name = self.df.image.iloc[idx]
        try:
           img = self.cache[img_name]
        except KeyError:
            img = self.loader(os.path.join(self.datapath, self.dirname, img_name+'.jpg'))
            self.cache[img_name] = img
        label = self._get_label(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)
