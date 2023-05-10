#!/usr/bin/env python
import os

from .base import CachedCSVDataset

ddi_map = {
        'acral-melanotic-macule': 'melanoma look-alike',
        'atypical-spindle-cell-nevus-of-reed': 'melanoma look-alike',
        'benign-keratosis': 'melanoma look-alike',
        'blue-nevus': 'melanoma look-alike',
        'dermatofibroma': 'melanoma look-alike',
        'dysplastic-nevus': 'melanoma look-alike',
        'epidermal-nevus': 'melanoma look-alike',
        'hyperpigmentation': 'melanoma look-alike',
        'keloid': 'melanoma look-alike',
        'inverted-follicular-keratosis': 'melanoma look-alike',
        'melanocytic-nevi': 'melanoma look-alike',
        'melanoma': 'melanoma',
        'melanoma-acral-lentiginous': 'melanoma',
        'melanoma-in-situ': 'melanoma',
        'nevus-lipomatosus-superficialis': 'melanoma look-alike',
        'nodular-melanoma-(nm)': 'melanoma',
        'pigmented-spindle-cell-nevus-of-reed': 'melanoma look-alike',
        'seborrheic-keratosis': 'melanoma look-alike',
        'seborrheic-keratosis-irritated': 'melanoma look-alike',
        'solar-lentigo': 'melanoma look-alike'
        }

class DDIDataset(CachedCSVDataset):
    """PyTorch dataset of Diverse Dermatology Images.

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
        self.datapath = "data/ddi/"
        self.csvname = "ddi_metadata.csv"
        self.mimics_only = mimics_only

        super().__init__(transform=transform)
        if self.mimics_only:
            mimic_kws = set(ddi_map.keys())
            self.df = self.df.query('disease in @mimic_kws')

    def _get_label(self, idx):
        text_label = self.df.disease.iloc[idx]
        try:
            label = (ddi_map[text_label] == 'melanoma')
        except KeyError:
            assert self.mimics_only == False
            label = False
        return label

    def __getitem__(self, idx):
        img_name = self.df.DDI_file.iloc[idx]
        try:
           img = self.cache[img_name]
        except KeyError:
            img = self.loader(os.path.join(self.datapath, img_name))
            self.cache[img_name] = img
        label = self._get_label(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)
