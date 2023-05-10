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

from utils.color import TemperatureShiftSRGB

def df_to_diff(df):
    temperatures = []
    diffs = []
    for col in df.columns[2:]:
        diff = df[col] - df['original']
        diffs.append(diff.mean())
        temperature = float(col)
        temperatures.append(temperature)
    return temperatures, diffs

def temperature_to_color(reference_temperature, test_temperature=6500):
    shift = TemperatureShiftSRGB(test_temperature, reference_temperature)
    t = shift(torch.ones(3).unsqueeze(0).unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(1).squeeze(1)
    t = t.detach().cpu().numpy()
    return t[0], t[1], t[2]

def main():
    matplotlib.rcParams['font.size'] = 7
    rotate_pct = 0.4
    fig, ax = plt.subplots()
    fig.set_size_inches(8.8/2.54, 6/2.54)
    data = [("deepderm", "temperature_shift_deepderm_isic.csv"),
            ("modelderm", "temperature_shift_modelderm_isic.csv"),
            ("scanoma", "temperature_shift_scanoma_isic.csv"),
            ("sscd", "temperature_shift_sscd_isic.csv")]

    ax.plot([0,30000],[0,0], color='black', lw=ax.spines['left'].get_linewidth(), ls='--')
    for i, (name, path) in enumerate(data):
        df = pandas.read_csv(path)
        temperatures, diffs = df_to_diff(df)
        xs = temperatures
        ys = np.array(diffs)
        line_color = (i/len(data), i/len(data), i/len(data))
        ax.semilogx(xs, ys, color=line_color, label=name)
    #for temperature in temperatures:
    #    color = temperature_to_color(temperature)
    #    ax.semilogx(temperature, -1.2, color=color, marker='s', clip_on=False)

    for temperature in np.linspace(min(temperatures), max(temperatures), 1000, dtype=np.float32):
        color = temperature_to_color(float(temperature))
        ax.semilogx(temperature,-1.5, color=color, marker='|', clip_on=False) 
    ax.semilogx([6500,6500],[-1,1], ls='dotted', color='black')

    ax.set_xlim(min(temperatures),max(temperatures))
    ax.set_ylim(-1,1)
    for kw in ['top', 'right', 'bottom']:
        ax.spines[kw].set_visible(False)
    xticks = [2000,4000,6000,8000,10000,15000,20000,25000]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["2", "4", "6", "8", "10", "15", "20", "25"])
    ax.set_xlabel("Correlated color temperature (\u00d710\u00b3 K)")
    ax.set_yticks([-1,0,1])
    ax.yaxis.set_minor_locator(ticker.FixedLocator([-.8,-.6,-.4,-.2,.2,.4,.6,.8]))
    ax.set_ylabel("Mean change in model output")
    fig.subplots_adjust(bottom=0.2)
    ax.legend()
    plt.savefig("temperature.pdf")


if __name__ == "__main__":
    main()
