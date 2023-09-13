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

from utils.color.cam16 import CAM16ToXYZ
from utils.color.cat import CAT16Torch

NAME_MAP = {
    'deepderm': 'DeepDerm',
    'modelderm': 'ModelDerm',
    'scanoma': 'Scanoma',
    'sscd': 'SSCD',
    'siimisic': 'SIIM-ISIC'
    }
TEXT_BUFFER = 0.05

def rotate(arr, pct):
    n = int(arr.shape[0]*pct)
    return np.concatenate((arr[n:], arr[:n]))

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
    data = [("deepderm",  "cat16_deepderm_isic.csv"),
            ("modelderm", "cat16_modelderm_isic.csv"),
            ("scanoma",   "cat16_scanoma_isic.csv"),
            ("sscd",      "cat16_sscd_isic.csv"),
            ("siim-isic", "cat16_siimisic_isic.csv")]

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
    C = 25
    J = 100
    C = torch.tensor(C, dtype=torch.float32)
    J = torch.tensor(J, dtype=torch.float32)
    xyz_wr = torch.tensor([95.04, 100, 108.88], dtype=torch.float32) # D65
    cam16_to_xyz = CAM16ToXYZ(xyz_wr, 300)
    for x in np.linspace(0,1,1000, dtype=np.float32):
        h = torch.tensor(x*360, dtype=torch.float32)
        xyz_w = cam16_to_xyz(J,C,h)
        color_shift = CAT16Torch(xyz_w, xyz_wr, D=1)
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
    plt.savefig("cat16.pdf")

if __name__ == "__main__":
    main()
