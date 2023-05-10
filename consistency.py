#!/usr/bin/env python
__doc__ = """Calculate the consistency between a generative explanation model
obtained via Explanation by Progressive Exaggeration and a set of target 
probabilities. For instance, a typical model may be trained to output images
that cause a classifier to predict the following probabilities P:

  Bin:   0  |   1  |   2  |   3  |   4  |   5  |   6  |   7  |   8  |   9
  P:   0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95

However, the classifier's actual prediction may differ. As a result, even if 
the extreme bins lie on opposite sides of the decision boundary, a generated
image may fail to flip the classifier's decision.

This script calculates the consistency between the target classifier outputs
and the achieved classifier outputs for each bin via mean squared error. 
In addition, this script calculates how often the model successfully flips
the classifier's output by comparing the classifier's prediction on the
original image to the classifier's prediction on the generated image from 
the extreme bin (0 or 9) on the opposite side of the decision boundary.
"""
import argparse
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from datasets import ISICDataset, Fitzpatrick17kDataset
from models import Generator
from models import DeepDermClassifier
from models import ModelDermClassifier
from models import ScanomaClassifier
from models import SSCDClassifier

DATASET_MAP = {
    "f17k": Fitzpatrick17kDataset,
    "isic": ISICDataset
    }
CLASSIFIER_MAP = {
    "deepderm": DeepDermClassifier,
    "modelderm": ModelDermClassifier,
    "scanoma": ScanomaClassifier,
    "sscd": SSCDClassifier
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, 
            help="The file path to the checkpoint file for the generative "
                 "model.")
    parser.add_argument("--dataset", type=str, choices=["f17k", "isic"],
            help="The dataset on which to evaluate the classifier and "
                 "generative model. Choose 'f17k' for Fitzpatrick7k or 'isic' "
                 "for ISIC 2019.")
    parser.add_argument("--classifier", type=str, 
            choices=["deepderm", "modelderm", "scanoma", "sscd"],
            help="The classifier against which to evaluate the generative"
                 "model.")
    parser.add_argument("--output_prefix", type=str,
            help="The classifier's predictions will be saved to two files: " 
                 "<output_prefix>_target.npy and <output_prefix>_original.npy. "
                 "The first contains a shape (n_classes, n_images) array of "
                 "the classifier's predictions on images generated from each "
                 "bin, while the second contains a shape (n_images) array of "
                 "the classifier's predictions on the original images.")
    parser.add_argument("--min_prob", type=float, default=0,
            help="This script assumes that there are 10 target outputs evenly "
                 "spaced between 'min_prob' and 'max_prob'. For instance, if "
                 "min_prob=0 and max_prob=1 (default), then the target "
                 "predictions are [0.05, 0.15, 0.25, ..., 0.95]")
    parser.add_argument("--max_prob", type=float, default=1,
            help="See --min_prob.")
    parser.add_argument("--threshold", type=float, default=0.5,
            help="The classifier's decision threshold.")
    parser.add_argument("--generator_image_size", type=int, default=224,
            help="The edge length, in pixels, of the images produced by the "
                 "generator (assumed square). Typically this will match the "
                 "input size of the classifier, but may not match if the "
                 "generative model was trained using a different classifier "
                 "than the classifier selected for evaluation.")
    args = parser.parse_args()
    dataset_class = DATASET_MAP[args.dataset]
    classifier_class = CLASSIFIER_MAP[args.classifier]
    checkpoint_path = args.checkpoint_path

    # Other defaults
    num_classes = 10
    device = 'cuda'
    batch_size = 8

    # Load checkpoint and classifier
    checkpoint = torch.load(checkpoint_path)
    classifier = classifier_class()
    generator_image_size = classfier.image_size if args.generator_image_size is None else args.generator_image_size
    generator = Generator(im_size=generator_image_size)
    generator.load_state_dict(checkpoint['generator'])
    classifier.to(device)
    generator.to(device)

    # Set up transforms; size is specific to generator
    normalize = transforms.Normalize(mean=0.5,
                                     std=0.5)
    transform = transforms.Compose([
            transforms.Resize(generator_image_size),
            transforms.CenterCrop(generator_image_size),
            transforms.ToTensor(),
            normalize])

    # Initialize dataset
    dataset = dataset_class(transform=transform)
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=4)
    
    # Iterate over images 
    target_preds = np.zeros((num_classes, len(dataset)))
    original_preds = np.zeros((len(dataset),))
    for ibatch, batch in enumerate(tqdm(dataloader)):
        img, label = batch 
        img = img.to(device)
        with torch.no_grad():
            for target_class in range(num_classes):
                # Transform image
                targets = target_class*torch.ones(img.shape[0], dtype=torch.long).to(device)
                generated_img, _ = generator(img, targets)

                # Resize image if necessary
                if classifier.image_size == generator_image_size:
                    generated_img_ = generated_img
                    img_ = img
                else:
                    generated_img_ = torch.nn.functional.interpolate(
                            generated_img, 
                            size=classifier.image_size, 
                            mode='bicubic')
                    img_ = torch.nn.functional.interpolate(
                            img, 
                            size=classifier.image_size, 
                            mode='bicubic')
                # Store classifier predictions
                target_pred = classifier(generated_img_)[:,classifier.positive_index]
                target_preds[target_class, ibatch*batch_size:(ibatch+1)*batch_size] = \
                            target_pred.detach().cpu().numpy()

            # Store classifier predictions for original image
            original_pred = classifier(img_)[:,classifier.positive_index]
            original_preds[ibatch*batch_size:(ibatch+1)*batch_size] = \
                       original_pred.detach().cpu().numpy()

    if args.output_prefix:
        np.save(args.output_prefix + "_target.npy", target_preds)
        np.save(args.output_prefix + "_original.npy", original_preds)

    # Calculate mean square error
    bins = np.linspace(args.min_prob, args.max_prob, num_classes+1)
    targets = np.array([bins[i:i+2].mean() for i in range(num_classes)])
    mse_list = []
    for i_class in range(num_classes):
        mse = ((target_preds[i_class]-targets[i_class])**2).sum()/target_preds.shape[1]
        mse_list.append(mse)
        print("MSE for target bin {:.03f}: {:f}".format(targets[i_class], mse))
    print("Mean MSE for all target classes: ", np.array(mse_list).mean())

    # Calculate true positive/negative rates for max/min bins
    tpr = (target_preds[-1]>args.threshold).sum()/target_preds.shape[1]
    tnr = (target_preds[0]<args.threshold).sum()/target_preds.shape[1]
    print("True positive rate for generator:", tpr)
    print("True negative rate for generator:", tnr)

    # Calculate flip rate
    positive_to_negative = target_preds[0, original_preds>args.threshold] < args.threshold
    print("Proportion flipped positive to negative: ", positive_to_negative.sum()/positive_to_negative.shape[0])
    negative_to_positive = target_preds[-1, original_preds<args.threshold] > args.threshold
    print("Proportion flipped negative to positive: ", negative_to_positive.sum()/negative_to_positive.shape[0])
    print("Proportion flipped overall:", (negative_to_positive.sum()+positive_to_negative.sum())/(negative_to_positive.shape[0]+positive_to_negative.shape[0]))

    # Calculate how often predictions move in correct direction
    positive_downward = target_preds[0, original_preds>args.threshold] < original_preds[original_preds>args.threshold]
    print("Proportion of originally positive images with prediction moved downward: ", 
          positive_downward.sum()/positive_downward.shape[0])
    negative_upward = target_preds[-1, original_preds<args.threshold] > original_preds[original_preds<args.threshold]
    print("Proportion of originally negative images with prediction moved upward: ", 
          negative_upward.sum()/negative_upward.shape[0])
    print("Proportion moved in correct direction overall",
          (negative_upward.sum() + positive_downward.sum())/(negative_upward.shape[0] + positive_downward.shape[0]))

if __name__ == "__main__":
    main()
