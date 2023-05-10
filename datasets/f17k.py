#!/usr/bin/env python
import os

from .base import CachedCSVDataset
from .f17k_exclusions import EXCLUSIONS

class Fitzpatrick17kDataset(CachedCSVDataset):
    """PyTorch dataset of Fitzpatrick17k images.

    The dataset caches images after the first load. If loading data using 
    multiple processes via a torch.utils.data.DataLoader instance, make sure 
    the worker processes are persistent (that is, use `persistent_workers=True`
    when initializing the DataLoader).

    Args:
        transform (nn.Module subclass): A torch module that transforms the 
            input. Typically, these transforms will be from torchvision.
        mimics_only (bool): Only include melanomas and melanoma mimickers."""
    def __init__(self, transform=None, mimics_only=True):
        self.datapath = "data/f17k/"
        self.csvname = "fitzpatrick17k.csv"
        self.dirname = "images_cropped"
        self.mimics_only = mimics_only

        super().__init__(transform=transform)
        # only consider certain conditions
        if self.mimics_only:
            self.df = self.df.query(
                    "nine_partition_label == 'malignant melanoma'"
                    " | nine_partition_label == 'benign melanocyte'"
                    " | label == 'seborrheic keratosis'"
                    " | label == 'dermatofibroma'")
            self.df = self.df.query("md5hash not in @EXCLUSIONS")

    def _get_label(self, idx):
        return self.df.nine_partition_label.iloc[idx] == 'malignant melanoma'

    def __getitem__(self, idx):
        img_name = self.df.md5hash.iloc[idx]
        try:
           img = self.cache[img_name]
        except KeyError:
            img = self.loader(os.path.join(self.datapath, self.dirname, img_name+'_crop.jpg'))
            self.cache[img_name] = img
        label = self._get_label(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)
