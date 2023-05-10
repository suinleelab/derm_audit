#!/usr/bin/env python
import argparse 

import pandas
import torch
from torchvision import transforms
from tqdm import tqdm

from models import DeepDermClassifier 
from models import ModelDermClassifier 
from models import ScanomaClassifier 
from models import SSCDClassifier 
from models import SIIMISICClassifier
import datasets

DEVICE = 'cuda'

OUTPATHS = {
        'scanoma':
            {'f17k': 'scanoma_f17k.csv',
             'isic': 'scanoma_isic.csv',
             'ddi': 'scanoma_ddi.csv'},
        'sscd':
            {'f17k': 'sscd_f17k.csv',
             'isic': 'sscd_isic.csv',
             'ddi': 'sscd_ddi.csv'},
        'deepderm':
            {'f17k': 'deepderm_f17k.csv',
             'isic': 'deepderm_isic.csv',
             'ddi': 'deepderm_ddi.csv'},
        'modelderm':
            {'f17k': 'modelderm_f17k.csv',
             'isic': 'modelderm_isic.csv',
             'ddi': 'modelderm_ddi.csv'},
        'siimisic':
            {'f17k': 'siimisic_f17k.csv',
             'isic': 'siimisic_isic.csv',
             'ddi': 'siimisic_ddi.csv'}
        }

CLASSIFIER_CLASS = {
        'scanoma': ScanomaClassifier,
        'sscd': SSCDClassifier,
        'deepderm': DeepDermClassifier,
        'modelderm': ModelDermClassifier,
        'siimisic': SIIMISICClassifier
        }

DATASET_CLASS = {
        'f17k': datasets.Fitzpatrick17kDataset,
        'isic': datasets.ISICDataset,
        'ddi': datasets.DDIDataset
        }

def basic_test(model_name, dataset_name):
    outpath = OUTPATHS[model_name][dataset_name]
    cls = CLASSIFIER_CLASS[model_name]
    dataset_class = DATASET_CLASS[dataset_name]
    classifier = CLASSIFIER_CLASS[model_name]()
    im_size = classifier.image_size

    positive_index = classifier.positive_index
    batch_size = 16

    normalize = transforms.Normalize(mean=0.5,
                                     std=0.5)
    transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            normalize])

    dataset = dataset_class(transform=transform)
    #dataset.df = dataset.df.iloc[:500]
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=1)

    classifier.eval()
    classifier.to(DEVICE)
    try: classifier.enable_augment()
    except AttributeError: pass

    labels = []
    preds = []
    for ibatch, batch in enumerate(tqdm(dataloader)):
        im, label = batch
        im = im.to(DEVICE)
        with torch.no_grad():
            pred = classifier(im)[:, positive_index].detach().cpu().numpy()
            labels += list(label.detach().cpu().numpy())
            preds += list(pred)

    d = {'ground_truth': labels, 'prediction': preds}
    if 'fitzpatrick' in dataset.df.columns:
        fitzpatrick_list = []
        for i in range(len(dataset)):
            fitzpatrick = dataset.df.fitzpatrick.iloc[i]
            label = dataset._get_label(i)
            assert label == labels[i]
            fitzpatrick_list.append(fitzpatrick)
        d['fitzpatrick'] = fitzpatrick_list
    if 'skin_tone' in dataset.df.columns:
        fitzpatrick_list = []
        for i in range(len(dataset)):
            skin_tone = dataset.df.skin_tone.iloc[i]
            label = dataset._get_label(i)
            assert label == labels[i]
            fitzpatrick_list.append(skin_tone)
        d['fitzpatrick'] = fitzpatrick_list

    df = pandas.DataFrame(d)
    df.to_csv(outpath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, choices=CLASSIFIER_CLASS.keys())
    parser.add_argument('--dataset', type=str, choices=DATASET_CLASS.keys())
    args = parser.parse_args()
    basic_test(args.classifier, args.dataset)

if __name__ == "__main__":
    main()
