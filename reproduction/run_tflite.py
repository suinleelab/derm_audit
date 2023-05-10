#!/usr/bin/env python
"""
THIS SCRIPT MUST BE RUN FROM ITS PARENT DIRECTORY!
ex: python reproduction/run_tflite.py

Evaluate tensorflow lite implementations of Scanoma and Smart Skin Cancer 
Detection models on images from the ISIC dataset.
"""
import os
import sys
# Frob the path. Apologies in advance...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas
from tqdm import tqdm

from datasets import ISICDataset
from reproduction.scanoma import TFLiteScanomaClassifier
from reproduction.sscd import TFLiteSSCDClassifier
from reproduction.sscd import load_and_preprocess_cropped

### User-editable constants. ###
SCANOMA_TFLITE_PATH = "/homes/gws/degrave/projects/derm/2021.08.09/scanoma/model.tflite"
SSCD_TFLITE_PATH = "/homes/gws/degrave/projects/derm/sscd.tflite"

# Additional constants that shouldn't need to be edited.
ISIC_IMG_PATH = "/fdata/derm/ISIC_2019_Training_Input"

SCANOMA_OUTPUT_PATH = "./reproduction/scanoma_tflite.csv"
SSCD_OUTPUT_PATH = "./reproduction/sscd_tflite.csv"

N_TEST_IMGS = 1000

SCANOMA_MEL_INDEX = 1
SSCD_MEL_INDEX = 1

def main():
    # Initialize models and dataset
    scanoma = TFLiteScanomaClassifier(SCANOMA_TFLITE_PATH)
    sscd = TFLiteSSCDClassifier(SSCD_TFLITE_PATH)
    ds = ISICDataset()

    # Run the evaluation
    scanoma_p = [] 
    sscd_p = []
    sscd_p_cropped = []
    paths = []
    for i_img in tqdm(range(N_TEST_IMGS)):
        # Get image path
        name = ds.df.iloc[i_img].image
        path = os.path.join(ISIC_IMG_PATH, name + ".jpg")
        paths.append(path)

        # Run classifiers
        scanoma_p.append(scanoma.run_from_filepath(path)[SCANOMA_MEL_INDEX])
        sscd_p.append(sscd.run_from_filepath(path)[SSCD_MEL_INDEX])
        cropped_img = load_and_preprocess_cropped(path)
        sscd_p_cropped.append(sscd.run(cropped_img)[SSCD_MEL_INDEX])

    # Save outputs to csv
    scanoma_df = pandas.DataFrame({"image": paths, "tflite_p": scanoma_p})
    scanoma_df.to_csv(SCANOMA_OUTPUT_PATH)

    sscd_df = pandas.DataFrame({"image": paths, 
                                "tflite_p": sscd_p, 
                                "tflite_p_cropped": sscd_p_cropped})
    sscd_df.to_csv(SSCD_OUTPUT_PATH)

if __name__ == "__main__":
    main()
