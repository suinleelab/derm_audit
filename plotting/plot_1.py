#!/usr/bin/env python
"""
Plot the overall predictive performance (ROC-AUC) of the classifiers.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm

color_scheme = {'good': '#5297bf',
                'bad': '#bf5b52',
                'unknown': '#aaaaaa'}

isic_paths = ['../2022.01.07/deepderm_isic.csv',
              '../2022.01.07/modelderm_isic.csv',
              '../2022.01.07/scanoma_isic.csv',
              '../2022.01.07/sscd_isic.csv',
              '../2022.06.01/siimisic_isic.csv']
f17k_paths = ['../2022.01.07/deepderm_f17k.csv',
              '../2022.01.07/modelderm_f17k.csv',
              '../2022.01.07/scanoma_f17k.csv',
              '../2022.01.07/sscd_f17k.csv',
              '../2022.06.01/siimisic_f17k.csv']
ddi_paths = ['../2022.06.01/deepderm_ddi.csv',
             '../2022.06.01/modelderm_ddi.csv',
             '../2022.06.01/scanoma_ddi.csv',
             '../2022.06.01/sscd_ddi.csv',
             '../2022.06.01/siimisic_ddi.csv']
model_names = ['DeepDerm',
               'ModelDerm 2018',
               'Scanoma',
               'SSCD',
               'SIIM-ISIC 2020']
isic_colors = [color_scheme['bad'], # deepderm
               color_scheme['good'], # modelderm
               color_scheme['unknown'], # scanoma
               color_scheme['unknown'], # sscd
               color_scheme['bad']] # siim-isic
f17k_colors = [color_scheme['bad'], # deepderm
               color_scheme['good'], # modelderm
               color_scheme['unknown'], # scanoma
               color_scheme['unknown'], # sscd
               color_scheme['good']] # siim-isic
ddi_colors = [color_scheme['good'], # deepderm
              color_scheme['good'], # modelderm
              color_scheme['good'], # scanoma
              color_scheme['good'], # sscd
              color_scheme['good']] # siim-isic


def roc_with_confidence(ground_truth, prediction, nsamples=1000, interval=0.95):
    rng = np.random.default_rng()
    rocs = []
    for i in tqdm(range(nsamples), leave=False):
        # Flip the ground_truth labels according to likelihood they are correct
        idxs = rng.integers(0, high=ground_truth.shape[0], size=ground_truth.shape[0])
        try:
            roc = sklearn.metrics.roc_auc_score(ground_truth.iloc[idxs], prediction.iloc[idxs])
            rocs.append(roc)
        except ValueError: # occurs when all the values in ground_truth are identical, which may occur by chance
            continue
    rocs.sort()
    n_tail = int(nsamples*interval/2)
    lb = rocs[n_tail]
    ub = rocs[-1*n_tail]
    #roc = sklearn.metrics.roc_auc_score(ground_truth, prediction)
    roc = np.median(rocs)
    return roc, (lb, ub)

def main():
    matplotlib.rcParams['font.size'] = 7
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(8.8/2.54, 6/2.54)

    xs = [0,1,2,3,4]
    plot_params = {'edgecolor': None,
                   'width': 0.5}
    # isic
    rocs = []
    yerrs_l = []
    yerrs_u = []
    for path in isic_paths:
        csv = pd.read_csv(path)
        roc, (lb, ub) = roc_with_confidence(csv['ground_truth'], csv['prediction'])
        rocs.append(roc)
        yerrs_l.append(roc-lb)
        yerrs_u.append(ub-roc)
    print(rocs, yerrs_l, yerrs_u)
    axs[0].bar(xs, rocs, yerr=(yerrs_l, yerrs_u), color=isic_colors, **plot_params)

    # f17k 
    rocs = []
    yerrs_l = []
    yerrs_u = []
    for path in f17k_paths:
        csv = pd.read_csv(path)
        roc, (lb, ub) = roc_with_confidence(csv['ground_truth'], csv['prediction'])
        rocs.append(roc)
        yerrs_l.append(roc-lb)
        yerrs_u.append(ub-roc)
    print(rocs, yerrs_l, yerrs_u)
    axs[1].bar(xs, rocs, yerr=(yerrs_l, yerrs_u), color=f17k_colors, **plot_params)

    # ddi
    rocs = []
    yerrs_l = []
    yerrs_u = []
    for path in ddi_paths:
        csv = pd.read_csv(path)
        roc, (lb, ub) = roc_with_confidence(csv['ground_truth'], csv['prediction'])
        rocs.append(roc)
        yerrs_l.append(roc-lb)
        yerrs_u.append(ub-roc)
    print(rocs, yerrs_l, yerrs_u)
    axs[2].bar(xs, rocs, yerr=(yerrs_l, yerrs_u), color=ddi_colors, **plot_params)

    for ax in (axs[0], axs[1], axs[2]):
        for kw in ['top', 'right']:
            ax.spines[kw].set_visible(False)
        ax.set_ylim(0.5,1)
        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
    axs[0].set_ylabel("ROC-AUC")
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    axs[0].set_title("ISIC 2019", fontsize=7)
    axs[1].set_title("Fitzpatrick17k", fontsize=7)
    axs[2].set_title("DDI", fontsize=7)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig("plot1.pdf")
    

if __name__ == "__main__":
    main()
