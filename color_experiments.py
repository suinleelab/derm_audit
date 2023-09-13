#!/usr/bin/env python
import pandas as pd
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
import utils.color
from utils.color.cam16 import CAM16ToXYZ
from utils.color.cielab import CIELABToCIEXYZ
from utils.color.cat import CIELUVAdaptationTorch
from utils.color.cat import CIELABAdaptationTorch
from utils.color.cat import CAT16Torch 

def cieluv_experiment(model_class, dataset_class, outpath, n_colors=32, rho=20, DEVICE='cuda', save_images=False):
    '''
    Examine the systematic effects of color modification via a translational 
    (Judd-type) chromatic adaptation in the CIELUV color space. White points 
    are selected uniformly from a circle about (u*,v*)=(0,0) with radius rho.

    Args:
      model_class: The uninitialized AI device (a class from the "models" 
        module).
      dataset_class: The uninitialized dataset (a class from the "datasets" 
        module)
      outpath: (str) The file path at which to save the output CSV file
      n_colors: (int) The number of different white points to test
      rho: (float) The radius of the circle in the u*, v*-plane about 
        (u*, v*)=(0,0) from which to choose white points.
      DEVICE: (str or torch device) The torch device on which to perform 
        calculations
      save_images: (bool) If True, save example images but do not perform 
        calculations
    '''
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
            color_shift = CIELUVAdaptationTorch(delta_u_star, delta_v_star)
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
    df = pd.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)


def cielab_experiment(model_class, dataset_class, outpath, n_colors=32, rho=20, DEVICE='cuda', save_images=False):
    '''
    Examine the systematic effects of color modification via a von Kries-like 
    chromatic adaptation as defined by the CIELAB color space. This is 
    equivalent to transforming from sRGB (white point D65) to CIELAB with a 
    newly chosen illuminant, then transforming back to sRGB with the D65 
    illuminant. White points are selected uniformly from a circle about 
    (a*,a*)=(0,0) with radius rho.

    Args:
      model_class: The uninitialized AI device (a class from the "models" 
        module).
      dataset_class: The uninitialized dataset (a class from the "datasets" 
        module)
      outpath: (str) The file path at which to save the output CSV file
      n_colors: (int) The number of different white points to test
      rho: (float) The radius of the circle in the a*, b*-plane about 
        (a*, b*)=(0,0) from which to choose white points.
      DEVICE: (str or torch device) The torch device on which to perform 
        calculations
      save_images: (bool) If True, save example images but do not perform 
        calculations
    '''
    classifier = model_class()
    positive_index = classifier.positive_index
    im_size = classifier.image_size
    batch_size = 64

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

        # Images with shifted a*, b*
        for i_color in range(n_colors):
            preds_current_color = []
            delta_a_star = rho*np.cos(i_color/n_colors*2*np.pi)
            delta_b_star = rho*np.sin(i_color/n_colors*2*np.pi)
            L = 100
            lab_n = torch.tensor([L, delta_a_star, delta_b_star]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            xyz_n = CIELABToCIEXYZ()(lab_n).squeeze()*100
            print(xyz_n)

            color_shift = CIELABAdaptationTorch(xyz_n) 
            color_shift.to(DEVICE)
            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("(Color {:02d} of {:02d})".format(i_color, n_colors))
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                shifted_im = color_shift(im)
                if save_images:
                    utils.color.save_torch(utils.color.torchvision_to_unit_interval(shifted_im[48].unsqueeze(0)), 
                                           #'cielab_images/{:.2f}_{:.2f}.jpg'.format(delta_a_star, delta_b_star))
                                           'cielab_images/{:02d}.jpg'.format(i_color))
                    break
                pred = classifier(shifted_im)[:, positive_index].detach().cpu().numpy()
                preds_current_color += list(pred)
            preds[repr((delta_a_star, delta_b_star))] = preds_current_color
    df = pd.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)

def cat16_experiment(model_class, dataset_class, outpath, n_colors=32, rho=30, DEVICE='cuda', save_images=False):
    '''
    Examine the systematic effects of color modification via the CAT16 
    (chromatic adaptation transform from CAM16). White points are selected 
    uniformly from a circle about (a,b)=(0,0) with radius rho.

    Args:
      model_class: The uninitialized AI device (a class from the "models" 
        module).
      dataset_class: The uninitialized dataset (a class from the "datasets" 
        module)
      outpath: (str) The file path at which to save the output CSV file
      n_colors: (int) The number of different white points to test
      rho: (float) The radius of the circle in the a, b-plane about 
        (a, b)=(0,0) from which to choose white points.
      DEVICE: (str or torch device) The torch device on which to perform 
        calculations
      save_images: (bool) If True, save example images but do not perform 
        calculations
    '''
    classifier = model_class()
    positive_index = classifier.positive_index
    im_size = classifier.image_size
    batch_size = 64

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

    # reference white
    xyz_wr = torch.tensor([95.04, 100, 108.88], dtype=torch.float32) # D65
    cam16_to_xyz = CAM16ToXYZ(xyz_wr, 300)

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

        # Images with shifted with white point adaptation 
        for i_color in range(n_colors):
            preds_current_color = []

            # CAM16
            h = i_color/n_colors*360
            C = rho
            J = 100
            xyz_w = cam16_to_xyz(J,C,h)

            color_shift = CAT16Torch(xyz_w, xyz_wr, D=1)
            color_shift.to(DEVICE)

            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("(Color {:02d} of {:02d})".format(i_color, n_colors))
            for ibatch, batch in enumerate(pbar):
                im, label = batch
                im = im.to(DEVICE)
                shifted_im = color_shift(im)
                if save_images:
                    utils.color.save_torch(utils.color.torchvision_to_unit_interval(shifted_im[48].unsqueeze(0)), 
                                           'cat16_images/{:02d}.jpg'.format(i_color))
                    break
                pred = classifier(shifted_im)[:, positive_index].detach().cpu().numpy()
                preds_current_color += list(pred)
            preds[repr((xyz_w[0], xyz_w[1], xyz_w[2]))] = preds_current_color
    df = pd.DataFrame.from_dict(preds)
    if not save_images:
        df.to_csv(outpath)


def main():
    cieluv_experiment(DeepDermClassifier, ISICDataset, "cieluv_deepderm_isic.csv")
    #cieluv_experiment(ModelDermClassifier, ISICDataset, "cieluv_modelderm_isic.csv")
    #cieluv_experiment(ScanomaClassifier, ISICDataset, "cieluv_scanoma_isic.csv")
    #cieluv_experiment(SSCDClassifier, ISICDataset, "cieluv_sscd_isic.csv")
    #cieluv_experiment(SIIMISICClassifier, ISICDataset, "cieluv_siimisic_isic.csv")

    cielab_experiment(DeepDermClassifier, ISICDataset, "cielab_deepderm_isic.csv")
    #cielab_experiment(ModelDermClassifier, ISICDataset, "cielab_modelderm_isic.csv")
    #cielab_experiment(ScanomaClassifier, ISICDataset, "cielab_scanoma_isic.csv")
    #cielab_experiment(SSCDClassifier, ISICDataset, "cielab_sscd_isic.csv")
    #cielab_experiment(SIIMISICClassifier, ISICDataset, "cielab_siimisic_isic.csv")

    cat16_experiment(DeepDermClassifier, ISICDataset, "cat16_deepderm_isic.csv")
    #cat16_experiment(ModelDermClassifier, ISICDataset, "cat16_modelderm_isic.csv")
    #cat16_experiment(ScanomaClassifier, ISICDataset, "cat16_scanoma_isic.csv")
    #cat16_experiment(SSCDClassifier, ISICDataset, "cat16_sscd_isic.csv")
    #cat16_experiment(SIIMISICClassifier, ISICDataset, "cat16_siimisic_isic.csv")

if __name__ == "__main__":
    main()
