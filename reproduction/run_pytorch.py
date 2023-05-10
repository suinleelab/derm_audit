#!/usr/bin/env python
"""
THIS SCRIPT MUST BE RUN FROM ITS PARENT DIRECTORY!
ex: python reproduction/run_pytorch.py

Evaluate the PyTorch versions of Scanoma and Smart Skin Cancer Detection.
"""
import os
import sys
# Frob the path. Apologies in advance...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas
import torch
from torchvision import transforms
from tqdm import tqdm

from models import ModelDermClassifier
from models import ScanomaClassifier
from models import SSCDClassifier
from datasets import ISICDataset

if not os.path.exists('pretrained_classifiers'):
    raise Exception("Unable to find pretrained models. You may need to "
                    "either (i) run prepare.sh first, or (ii) run this "
                    "script from the root directory of the repository (not "
                    "the 'reproduction' subdirectory).")


### User-editable constants. ###
ISIC_CSV_PATH = "/fdata/derm/ISIC_2019_Training_GroundTruth.csv"
ISIC_IMG_PATH = "/fdata/derm/ISIC_2019_Training_Input"

# Additional constants that shouldn't need to be edited.
SCANOMA_OUTPUT_PATH = "reproduction/scanoma_pytorch.csv"
SSCD_OUTPUT_PATH = "reproduction/sscd_pytorch.csv"
MODELDERM_OUTPUT_PATH = "reproduction/modelderm_pytorch.csv"

N_TEST_IMGS = 1000
IM_SIZE = 224

SCANOMA_MEL_INDEX = 1
SSCD_MEL_INDEX = 1
MODELDERM_MEL_INDEX = 140
DEVICE = 'cuda'

def main():
    # Initialize models
    modelderm = ModelDermClassifier()
    scanoma = ScanomaClassifier()
    sscd = SSCDClassifier()

    modelderm.eval()
    scanoma.eval()
    sscd.eval()
    modelderm.to(DEVICE)
    scanoma.to(DEVICE)
    sscd.to(DEVICE)

    # Transforms for dataset
    normalize = transforms.Normalize(mean=0.5,
                                     std=0.5)
    # Standard transforms with bilinear resizing
    transform = transforms.Compose([
            transforms.Resize(IM_SIZE),
            transforms.CenterCrop(IM_SIZE),
            transforms.ToTensor(),
            normalize])

    # Transforms with nearest-neighbor resizing
    transform_nn = transforms.Compose([
            transforms.Resize(IM_SIZE, 
                     interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(IM_SIZE),
            transforms.ToTensor(),
            normalize])

    # Initialize dataset
    ds = ISICDataset(transform=transform)
    ds_nn = ISICDataset(transform=transform_nn)

    # Restrict dataset to first N_TEST_IMGS images
    ds.df = ds.df.iloc[:N_TEST_IMGS]
    ds_nn.df = ds_nn.df.iloc[:N_TEST_IMGS]

    # Run the evaluation
    modelderm_p = []
    scanoma_p = [] 
    sscd_p = []
    sscd_p_nn = []
    paths = []
    for i_img in tqdm(range(N_TEST_IMGS)):
        # Get image path
        name = ds.df.iloc[i_img].image
        path = os.path.join(ISIC_IMG_PATH, name + ".jpg")
        paths.append(path)

        # Evaluate with bilinear resizng
        image, label = ds[i_img]
        image = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            modelderm_p.append(modelderm(image).detach().cpu().numpy().squeeze()[MODELDERM_MEL_INDEX])
            scanoma_p.append(scanoma(image).detach().cpu().numpy().squeeze()[SCANOMA_MEL_INDEX])
            sscd_p.append(sscd(image).detach().cpu().numpy().squeeze()[SSCD_MEL_INDEX])

        # Evaluate SSCD with nearest-neighbor resizing
        image_nn, label_nn = ds_nn[i_img]
        image_nn = image_nn.unsqueeze(0).to(DEVICE)
        sscd_p_nn.append(sscd(image_nn).detach().cpu().numpy().squeeze()[SSCD_MEL_INDEX])

    # Save outputs to csv
    modelderm_df = pandas.DataFrame({"image": paths, "pytorch_p": modelderm_p})
    modelderm_df.to_csv(MODELDERM_OUTPUT_PATH)

    scanoma_df = pandas.DataFrame({"image": paths, "pytorch_p": scanoma_p})
    scanoma_df.to_csv(SCANOMA_OUTPUT_PATH)

    sscd_df = pandas.DataFrame({"image": paths, "pytorch_p": sscd_p, "pytorch_p_nn": sscd_p_nn})
    sscd_df.to_csv(SSCD_OUTPUT_PATH)

if __name__ == "__main__":
    main()
