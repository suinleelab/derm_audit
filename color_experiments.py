#!/usr/bin/env python
import pandas
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from models import DeepDermClassifier 
from models import ModelDermClassifier 
from models import ScanomaClassifier 
from models import SSCDClassifier 
from models import SIIMISICClassifier
from datasets import ISICDataset
from utils.color import ChromaticityShiftTorch
from utils.color import TemperatureShiftTorch
import utils.color

def chromaticity_shift_experiment(model_class, dataset_class, outpath, n_colors=32, rho=20, DEVICE='cuda', save_images=False):
    classifier = model_class()
    positive_index = classifier.positive_index
    im_size = classifier.image_size
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
            num_workers=1,
            pin_memory=True)

    classifier.eval()
    classifier.to(DEVICE)
    try: classifier.enable_augment()
    except AttributeError: pass

    preds = {}

    with torch.no_grad():
        # Do original images first
        pbar = tqdm(dataloader, leave=False)
        pbar.set_description("(Original color)")
        preds_original = []
        if not save_images:
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                pred = classifier(im)[:, positive_index].detach().cpu().numpy()
                preds_original += list(pred)
            preds['original'] = preds_original

        # Images with shifted u*, v*
        for i_color in range(n_colors):
            preds_current_color = []
            delta_u_star = rho*np.cos(i_color/n_colors*2*np.pi)
            delta_v_star = rho*np.sin(i_color/n_colors*2*np.pi)
            color_shift = ChromaticityShiftTorch(delta_u_star, delta_v_star)
            color_shift.to(DEVICE)
            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("(Color {:02d} of {:02d})".format(i_color, n_colors))
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                shifted_im = color_shift(im)
                if save_images:
                    utils.color.save_torch(utils.color.torchvision_to_unit_interval(shifted_im[0].unsqueeze(0)), 
                                           'color_shifted_images/{:.2f}_{:.2f}.jpg'.format(delta_u_star, delta_v_star))
                    break
                pred = classifier(shifted_im)[:, positive_index].detach().cpu().numpy()
                preds_current_color += list(pred)
            preds[repr((delta_u_star, delta_v_star))] = preds_current_color
    df = pandas.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)

def temperature_shift_experiment(model_class, dataset_class, outpath, n_colors=32, DEVICE='cuda', save_images=False):
    temperatures = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000, 10000, 15000, 20000, 25000] 
    classifier = model_class()
    positive_index = classifier.positive_index
    im_size = classifier.image_size
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
            num_workers=1,
            pin_memory=True)

    classifier.eval()
    classifier.to(DEVICE)
    try: classifier.enable_augment()
    except AttributeError: pass

    preds = {}

    with torch.no_grad():
        # Do original images first
        pbar = tqdm(dataloader, leave=False)
        pbar.set_description("(Original color)")
        preds_original = []
        if not save_images:
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                pred = classifier(im)[:, positive_index].detach().cpu().numpy()
                preds_original += list(pred)
            preds['original'] = preds_original

        # Images with shifted temperature
        for i_color, reference_temperature in enumerate(temperatures):
            preds_current_color = []
            test_temperature = 6500
            color_shift = TemperatureShiftTorch(test_temperature, reference_temperature)
            color_shift.to(DEVICE)
            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("(Color {:02d} of {:02d})".format(i_color, len(temperatures)))
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                shifted_im = color_shift(im)
                if save_images:
                    utils.color.save_torch(utils.color.torchvision_to_unit_interval(shifted_im[0].unsqueeze(0)), 
                            'color_shifted_images/{:f}.jpg'.format(reference_temperature))
                    break
                pred = classifier(shifted_im)[:, positive_index].detach().cpu().numpy()
                preds_current_color += list(pred)
            preds[reference_temperature] = preds_current_color
    df = pandas.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)

def main():
    chromaticity_shift_experiment(DeepDermClassifier, ISICDataset, "color_shift_deepderm_isic.csv")
    chromaticity_shift_experiment(ModelDermClassifier, ISICDataset, "color_shift_modelderm_isic.csv")
    chromaticity_shift_experiment(ScanomaClassifier, ISICDataset, "color_shift_scanoma_isic.csv")
    chromaticity_shift_experiment(SSCDClassifier, ISICDataset, "color_shift_sscd_isic.csv")
    chromaticity_shift_experiment(SIIMISICClassifier, ISICDataset, "color_shift_siimisic_isic.csv")
    temperature_shift_experiment(DeepDermClassifier, ISICDataset, "temperature_shift_deepderm_isic.csv")
    temperature_shift_experiment(ModelDermClassifier, ISICDataset, "temperature_shift_modelderm_isic.csv")
    temperature_shift_experiment(ScanomaClassifier, ISICDataset, "temperature_shift_scanoma_isic.csv")
    temperature_shift_experiment(SSCDClassifier, ISICDataset, "temperature_shift_sscd_isic.csv")
    temperature_shift_experiment(SIIMISICClassifier, ISICDataset, "temperature_shift_siimisic_isic.csv")

if __name__ == "__main__":
    main()
