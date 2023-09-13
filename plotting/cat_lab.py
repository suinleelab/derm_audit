#!/usr/bin/env python
'''
Plot the response of each classifier to changing chromaticity.
'''
import pandas
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from utils.color.cat import CIELABAdaptationTorch
from utils.color.cielab import CIELABToCIEXYZ

NAME_MAP = {
    'deepderm': 'DeepDerm',
    'modelderm': 'ModelDerm',
    'scanoma': 'Scanoma',
    'sscd': 'SSCD',
    'siimisic': 'SIIM-ISIC'
    }
TEXT_BUFFER = 0.05

def df_to_diff(df):
    diffs = []
    for col in df.columns[2:]:
        diff = df[col] - df['original']
        diffs.append(diff.mean())
    return diffs

def rotate_ys(ys, degree=0.5):
    cutoff = int(len(ys)*degree)
    return np.concatenate([ys[cutoff:], ys[:cutoff]])

def rotate_xs(xs, degree=0.5):
    result = xs - degree
    if isinstance(result, np.ndarray):
        result[result < 0] = result[result < 0] + 1
    elif isinstance(result, float):
        if result < 0:
            result += 1
    return result

def main():
    rho = 100
    matplotlib.rcParams['font.size'] = 7
    rotate_degree = 0
    fig, ax = plt.subplots()
    fig.set_size_inches(8.8/2.54, 6/2.54)
    data = [("deepderm",  "cielab_deepderm_isic.csv"),
            ("modelderm", "cielab_modelderm_isic.csv"),
            ("scanoma",   "cielab_scanoma_isic.csv"),
            ("sscd",      "cielab_sscd_isic.csv"),
            ("siim-isic", "cielab_siimisic_isic.csv")]

    ax.plot([0,1],[0,0], color='black', lw=ax.spines['left'].get_linewidth(), ls='--')
    
    max_ = 0
    for i, (name, path) in enumerate(data):
        df = pandas.read_csv(path)
        diffs = df_to_diff(df)
        ys = np.array(diffs)
        if np.abs(ys).max() > max_:
            max_ = np.abs(ys).max()
    print(max_)

    for i, (name, path) in enumerate(data):
        df = pandas.read_csv(path)
        diffs = df_to_diff(df)
        xs = np.linspace(0,1,len(diffs))
        ys = np.array(diffs)
        ys /= max_

        line_color = (i/len(data), i/len(data), i/len(data))
        ax.plot(xs, rotate_ys(ys, rotate_degree), color=line_color, label=name)

    rho = 25
    L = 100
    L = torch.tensor(L, dtype=torch.float32)
    n_colors = 1000
    for x in np.linspace(0,1,n_colors, dtype=np.float32):
        delta_a_star = rho*np.cos(x*2*np.pi)
        delta_b_star = rho*np.sin(x*2*np.pi)
        lab_n = torch.tensor([L, delta_a_star, delta_b_star]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        xyz_n = CIELABToCIEXYZ()(lab_n).squeeze()*100
        color_shift = CIELABAdaptationTorch(xyz_n) 
        color = color_shift(torch.tensor([0.5,0.5,0.5]).unsqueeze(0).unsqueeze(2).unsqueeze(3)).squeeze()
        ax.plot(rotate_xs(x, rotate_degree),-1.2, color=color.detach().numpy(), marker='|', clip_on=False) 

    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    for kw in ['top', 'right', 'bottom']:
        ax.spines[kw].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([-1,0,1])
    ax.yaxis.set_minor_locator(ticker.FixedLocator([-.8,-.6,-.4,-.2,.2,.4,.6,.8]))
    ax.set_ylabel("Mean change in model output")
    ax.legend()
    plt.savefig("cat_lab.pdf")

if __name__ == "__main__":
    main()
