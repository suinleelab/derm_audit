#!/usr/bin/env python 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

def df_to_diff(df):
    shifts = []
    diffs = []
    for col in df.columns[1:]:
        diff = df[col] - df['0.0']
        diffs.append(diff.mean())
        n = float(col)
        shifts.append(n)
    return shifts, np.array(diffs)

def plot_brightness_rgb():
    matplotlib.rcParams['font.size'] = 7
    fig, ax = plt.subplots()
    fig.set_size_inches(8.8/2.54, 6/2.54)
    data = [("deepderm", "brightness_rgb_shift_deepderm_f17k.csv"),
            ("modelderm", "brightness_rgb_shift_modelderm_f17k.csv"),
            ("scanoma", "brightness_rgb_shift_scanoma_f17k.csv"),
            ("sscd", "brightness_rgb_shift_sscd_f17k.csv"),
            ("siim-isic", "brightness_rgb_shift_siimisic_f17k.csv")]

    ax.plot([0,1],[0,0], color='black', lw=ax.spines['left'].get_linewidth(), ls='--')
    max_ = 0
    for i, (name, path) in enumerate(data):
        df = pd.read_csv(path)
        shifts, diffs = df_to_diff(df)
        if np.abs(diffs).max() > max_: max_ = np.abs(diffs).max()
    print(max_)
    for i, (name, path) in enumerate(data):
        df = pd.read_csv(path)
        shifts, diffs = df_to_diff(df)
        #xs = np.linspace(-0.5,0.5,len(diffs))
        xs = shifts
        ys = np.array(diffs/max_)
        line_color = (i/len(data), i/len(data), i/len(data))
        ax.plot(xs, ys, color=line_color, label=name)

    ax.set_xlim(-2,2)
    ax.set_ylim(-1,1)
    for kw in ['top', 'right', 'bottom']:
        ax.spines[kw].set_visible(False)
    ax.set_xticks([-2,-1,0,1,2])
    ax.set_yticks([-1,0,1])
    ax.yaxis.set_minor_locator(ticker.FixedLocator([-.8,-.6,-.4,-.2,.2,.4,.6,.8]))
    ax.set_ylabel("Mean change in model output")
    ax.legend()
    plt.savefig("brightness_rgb_f17k.pdf")

if __name__ == "__main__":
    plot_brightness_rgb()
