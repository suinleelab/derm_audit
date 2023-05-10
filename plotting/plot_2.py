#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm

from plotting.plot_1 import color_scheme
from plotting.plot_1 import f17k_paths
from plotting.plot_1 import ddi_paths
from plotting.plot_1 import model_names
from plotting.plot_1 import f17k_colors 
from plotting.plot_1 import ddi_colors 
from plotting.plot_1 import roc_with_confidence

matplotlib.rcParams['font.size'] = 7

def main():
    fig, axs = plt.subplots(2, len(model_names))
    fig.set_size_inches(18/2.54, 10/2.54)

    xs = [0,1,2]
    x_tick_labels = ['1-2', '3-4', '5-6']
    plot_params = {'edgecolor': None,
                   'width': 0.5}

    for i_dataset, (dataset_name, dataset_paths, dataset_colors) in \
            enumerate(zip(['DDI', 'Fitzpatrick17k'],
                          [ddi_paths, f17k_paths], 
                          [ddi_colors, f17k_colors]
                          )
                      ):
        for i_model, (model_name, model_path, model_color) in \
                enumerate(zip(model_names, dataset_paths, dataset_colors)):
            ax = axs[i_dataset, i_model]
            df = pd.read_csv(model_path)
            rocs = []
            yerrs_l = []
            yerrs_u = []
            print(dataset_name, model_name)
            for fitz_types in ([1,2,12], [3,4,34], [5,6,56]):
                subset = df.query("fitzpatrick in @fitz_types")
                roc, (lb, ub) = roc_with_confidence(subset['ground_truth'], subset['prediction'])
                rocs.append(roc)
                yerrs_l.append(roc-lb)
                yerrs_u.append(ub-roc)
            print(rocs, yerrs_l, yerrs_u)
            ax.bar(xs, rocs, yerr=(yerrs_l, yerrs_u), color=model_color, **plot_params)

            # Formatting
            for kw in ['top', 'right']:
                ax.spines[kw].set_visible(False)
            if i_model == 0:
                ax.set_ylabel(dataset_name)
                ax.set_yticks([0.5,0.6,0.7,0.8,0.9,1])
                ax.set_yticklabels(['0.5','','','','','1.0'])
            else:
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
            ax.set_xticks(xs)
            ax.set_xticklabels(x_tick_labels)
            ax.set_ylim(0.5, 1)
            if i_dataset == 0:
                ax.set_title(model_name)
    fig.subplots_adjust(hspace=0.2)
    plt.savefig("plot_2.pdf")

if __name__ == "__main__":
    main()
