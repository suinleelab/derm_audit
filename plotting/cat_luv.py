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

from utils.color.cieluv import CIELUVToCIEXYZ
from utils.color.ciexyz import CIEXYZToSRGB

def uv_to_color(u, v, factor=10):
    t = torch.tensor([50,u*factor,v*factor]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    xyz = CIELUVToCIEXYZ()(t)
    srgb = CIEXYZToSRGB()(xyz)
    srgb = srgb.squeeze(0).squeeze(1).squeeze(1).cpu().numpy()
    return srgb[0], srgb[1], srgb[2]

def df_to_diff(df):
    colors = []
    diffs = []
    for col in df.columns[2:]:
        diff = df[col] - df['original']
        diffs.append(diff.mean())
        uv = eval(col)
        color = uv_to_color(uv[0], uv[1])
        colors.append(color)
    return colors, diffs

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
    rotate_degree = 0.5
    fig, ax = plt.subplots()
    fig.set_size_inches(8.8/2.54, 6/2.54)
    data = [("deepderm", "cieluv_deepderm_isic.csv"),
            ("modelderm", "cieluv_modelderm_isic.csv"),
            ("scanoma", "cieluv_scanoma_isic.csv"),
            ("sscd", "cieluv_sscd_isic.csv"),
            ("siim-isic", "cieluv_siimisic_isic.csv")]

    ax.plot([0,1],[0,0], color='black', lw=ax.spines['left'].get_linewidth(), ls='--')

    max_ = 0
    for i, (name, path) in enumerate(data):
        df = pandas.read_csv(path)
        _, diffs = df_to_diff(df)
        ys = np.array(diffs)
        if np.abs(ys).max() > max_:
            max_ = np.abs(ys).max()
    print(max_)

    for i, (name, path) in enumerate(data):
        df = pandas.read_csv(path)
        colors, diffs = df_to_diff(df)
        xs = np.linspace(0,1,len(diffs))
        ys = np.array(diffs)
        ys /= max_

        line_color = (i/len(data), i/len(data), i/len(data))
        ax.plot(xs, rotate_ys(ys, degree=rotate_degree), color=line_color, label=name)
    # For testing alignment of discretized colors with nicer-looking gradient
    #for x, color in zip(xs, colors):
    #    ax.plot(x, -1.2, color=color, marker='s', clip_on=False)

    for x in np.linspace(0,1,1000, dtype=np.float32):
        delta_u = float(np.cos(x*2*np.pi))
        delta_v = float(np.sin(x*2*np.pi))
        color = uv_to_color(delta_u, delta_v, factor=rho)
        ax.plot(rotate_xs(x, degree=rotate_degree), -1.2, color=color, marker='|', clip_on=False) 

    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    for kw in ['top', 'right', 'bottom']:
        ax.spines[kw].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([-1,0,1])
    ax.yaxis.set_minor_locator(ticker.FixedLocator([-.8,-.6,-.4,-.2,.2,.4,.6,.8]))
    ax.set_ylabel("Mean change in model output")
    ax.legend()
    plt.savefig("cat_luv.pdf")


if __name__ == "__main__":
    main()
