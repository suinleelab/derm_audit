#!/usr/bin/env python
"""
THIS SCRIPT MUST BE RUN FROM ITS PARENT DIRECTORY!
ex: python reproduction/run_caffe.py

Evaluate PyCaffe implementation of ModelDerm 2018 on images from the ISIC 
dataset.
"""
import os
import sys
# Frob the path. Apologies in advance...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas
from tqdm import tqdm

from datasets import ISICDataset
from reproduction.modelderm import CaffeModelDermClassifier
from reproduction.modelderm import pytorch_load

# Constants that shouldn't need to be edited.
PROTO_PATH = "./pretrained_classifiers/deploy.prototxt"
MODEL_PATH = "./pretrained_classifiers/70616.caffemodel"
MEAN_PATH = "./pretrained_classifiers/mean224x224.binaryproto"

MODELDERM_OUTPUT_PATH = "reproduction/modelderm_caffe.csv"

ISIC_IMG_PATH = "/fdata/derm/ISIC_2019_Training_Input"

N_TEST_IMGS = 1000
IM_SIZE = 224
MEL_INDEX = 140 # based on ModelDerm 2018 label list

def main():
    # Load model
    model = CaffeModelDermClassifier(proto_path=PROTO_PATH,
                                     weight_path=MODEL_PATH,
                                     mean_path=MEAN_PATH)

    # Initialize dataset
    ds = ISICDataset()

    # Run the evaluation
    modelderm_p = [] 
    modelderm_p_warp = []
    paths = []
    for i_img in tqdm(range(N_TEST_IMGS)):
        # Get image path
        name = ds.df.iloc[i_img].image
        path = os.path.join(ISIC_IMG_PATH, name + ".jpg")
        paths.append(path)

        # Run classifier
        modelderm_p.append(model.run(pytorch_load(path, model.mean_image))[MEL_INDEX])
        #modelderm_p_warp.append(model.run_from_filepath(path)[MEL_INDEX])
    modelderm_df = pandas.DataFrame({"image": paths, 
                                     "caffe_p": modelderm_p})
                                     #"caffe_p": modelderm_p, 
                                     #"caffe_p_warp": modelderm_p_warp})
    modelderm_df.to_csv(MODELDERM_OUTPUT_PATH)

if __name__ == "__main__":
    main()
