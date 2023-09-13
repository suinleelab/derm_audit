#!/usr/bin/env python
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

from models import DeepDermClassifier 
from models import ModelDermClassifier 
from models import ScanomaClassifier 
from models import SSCDClassifier 
from models import SIIMISICClassifier
from datasets import Fitzpatrick17kDataset
from utils.color.brightness import LightnessShiftTorch
from utils.color.brightness import BrightnessRGBShiftTorch
from utils.color.brightness import JzScaleTorch

def brightness_rgb_shift_experiment(model_class, dataset_class, outpath, min_shift=-2, max_shift=2, n_colors=17, DEVICE='cuda', save_images=False):
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
    dataset.df = dataset.df[:100]
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
        for i_color in range(n_colors):
            preds_current_color = []
            brightness_factor = (max_shift-min_shift)/(n_colors-1)*i_color+min_shift
            color_shift = BrightnessRGBShiftTorch(brightness_factor)
            color_shift.to(DEVICE)
            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("(Color {:02d} of {:02d})".format(i_color, n_colors))
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                shifted_im = color_shift(im)
                if save_images:
                    utils.color.save_torch(utils.color.torchvision_to_unit_interval(shifted_im[5].unsqueeze(0)), 
                                           'brightnessRGB_shifted_images/{:.2f}.jpg'.format(brightness_factor))
                    break
                pred = classifier(shifted_im)[:, positive_index].detach().cpu().numpy()
                preds_current_color += list(pred)
            preds[repr(brightness_factor)] = preds_current_color
    df = pd.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)

def lightness_shift_experiment(model_class, dataset_class, outpath, min_shift=-15, max_shift=15, n_colors=17, DEVICE='cuda', save_images=False):
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
    dataset.df = dataset.df[:100]
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
        # Images with shifted L*
        for i_color in range(n_colors):
            preds_current_color = []
            delta_L_star = (max_shift-min_shift)/(n_colors-1)*i_color+min_shift
            color_shift = LightnessShiftTorch(delta_L_star)
            color_shift.to(DEVICE)
            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("(Color {:02d} of {:02d})".format(i_color, n_colors))
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                shifted_im = color_shift(im)
                if save_images:
                    utils.color.save_torch(utils.color.torchvision_to_unit_interval(shifted_im[5].unsqueeze(0)), 
                                           'lightness_shifted_images/{:.2f}.jpg'.format(delta_L_star))
                    break
                pred = classifier(shifted_im)[:, positive_index].detach().cpu().numpy()
                preds_current_color += list(pred)
            preds[repr(delta_L_star)] = preds_current_color
    df = pd.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)

def jz_shift_experiment(model_class, dataset_class, outpath, min_shift=-1, max_shift=1, n_colors=17, DEVICE='cuda', save_images=False):
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
    dataset.df = dataset.df[:100]
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
        # Images with shifted Jz
        for i_color in range(n_colors):
            preds_current_color = []
            delta_Jz = (max_shift-min_shift)/(n_colors-1)*i_color+min_shift
            color_shift = JzScaleTorch(delta_Jz)
            color_shift.to(DEVICE)
            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("(Color {:02d} of {:02d})".format(i_color, n_colors))
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                shifted_im = color_shift(im)
                if save_images:
                    utils.color.save_torch(utils.color.torchvision_to_unit_interval(shifted_im[5].unsqueeze(0)), 
                                           'jz_shifted_images/{:.4f}.jpg'.format(delta_Jz))
                    break
                pred = classifier(shifted_im)[:, positive_index].detach().cpu().numpy()
                preds_current_color += list(pred)
            preds[repr(delta_Jz)] = preds_current_color
    df = pd.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)

def main():
    brightness_rgb_shift_experiment(DeepDermClassifier, Fitzpatrick17kDataset, "brightness_rgb_shift_deepderm_f17k.csv")
    #brightness_rgb_shift_experiment(ModelDermClassifier, Fitzpatrick17kDataset, "brightness_rgb_shift_modelderm_f17k.csv")
    #brightness_rgb_shift_experiment(ScanomaClassifier, Fitzpatrick17kDataset, "brightness_rgb_shift_scanoma_f17k.csv")
    #brightness_rgb_shift_experiment(SSCDClassifier, Fitzpatrick17kDataset, "brightness_rgb_shift_sscd_f17k.csv")
    #brightness_rgb_shift_experiment(SIIMISICClassifier, Fitzpatrick17kDataset, "brightness_rgb_shift_siimisic_f17k.csv")

    lightness_shift_experiment(DeepDermClassifier, Fitzpatrick17kDataset, "lightness_shift_deepderm_f17k.csv")
    #lightness_shift_experiment(ModelDermClassifier, Fitzpatrick17kDataset, "lightness_shift_modelderm_f17k.csv")
    #lightness_shift_experiment(ScanomaClassifier, Fitzpatrick17kDataset, "lightness_shift_scanoma_f17k.csv")
    #lightness_shift_experiment(SSCDClassifier, Fitzpatrick17kDataset, "lightness_shift_sscd_f17k.csv")
    #lightness_shift_experiment(SIIMISICClassifier, Fitzpatrick17kDataset, "lightness_shift_siimisic_f17k.csv")

    jz_shift_experiment(DeepDermClassifier, Fitzpatrick17kDataset, "jz_shift_deepderm_f17k.csv")
    #jz_shift_experiment(ModelDermClassifier, Fitzpatrick17kDataset, "jz_shift_modelderm_f17k.csv")
    #jz_shift_experiment(ScanomaClassifier, Fitzpatrick17kDataset, "jz_shift_scanoma_f17k.csv")
    #jz_shift_experiment(SSCDClassifier, Fitzpatrick17kDataset, "jz_shift_sscd_f17k.csv")
    #jz_shift_experiment(SIIMISICClassifier, Fitzpatrick17kDataset, "jz_shift_siimisic_f17k.csv")

if __name__ == "__main__":
    main()
