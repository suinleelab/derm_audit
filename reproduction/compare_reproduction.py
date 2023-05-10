#!/usr/bin/env python
import pandas
import numpy as np
import matplotlib.pyplot as plt

MODELDERM_PYTORCH_CSV = "modelderm_pytorch.csv"
MODELDERM_CAFFE_CSV = "modelderm_caffe.csv"

SCANOMA_PYTORCH_CSV = "scanoma_pytorch.csv"
SCANOMA_TFLITE_CSV = "scanoma_tflite.csv"

SSCD_PYTORCH_CSV = "sscd_pytorch.csv"
SSCD_TFLITE_CSV = "sscd_tflite.csv"

def logit(p):
    """Given the probability, return base e logits."""
    return np.log(p/(1-p))

def plot(fig, xs, ys, savepath, min_=-5, max_=5):
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, '.', color='black')
    ax.plot([min_,max_],[min_,max_], '--', color='gray')
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
    ax.set_aspect('equal')
    ax.set_xlabel('Tensorflow Lite')
    ax.set_ylabel('PyTorch')
    plt.savefig(savepath)

def main():
    scanoma_pytorch_df = pandas.read_csv(SCANOMA_PYTORCH_CSV)
    scanoma_tflite_df = pandas.read_csv(SCANOMA_TFLITE_CSV)
    sscd_pytorch_df = pandas.read_csv(SSCD_PYTORCH_CSV)
    sscd_tflite_df = pandas.read_csv(SSCD_TFLITE_CSV)
    modelderm_pytorch_df = pandas.read_csv(MODELDERM_PYTORCH_CSV)
    modelderm_caffe_df = pandas.read_csv(MODELDERM_CAFFE_CSV)


    modelderm_corr = np.corrcoef(modelderm_pytorch_df['pytorch_p'], 
                               modelderm_caffe_df['caffe_p'])
    print("ModelDerm 2018:", modelderm_corr[0,1])

    scanoma_corr = np.corrcoef(scanoma_pytorch_df['pytorch_p'], 
                               scanoma_tflite_df['tflite_p'])
    print("Scanoma:", scanoma_corr[0,1])

    sscd_corr = np.corrcoef(sscd_pytorch_df['pytorch_p'], 
                            sscd_tflite_df['tflite_p'])
    print("SSCD (warping):", sscd_corr[0,1])

    sscd_corr2 = np.corrcoef(sscd_pytorch_df['pytorch_p'], 
                             sscd_tflite_df['tflite_p_cropped'])
    print("SSCD (cropping):", sscd_corr2[0,1])

    sscd_corr3 = np.corrcoef(sscd_pytorch_df['pytorch_p_nn'], 
                             sscd_tflite_df['tflite_p_cropped'])
    print("SSCD (cropping, nearest_neightbors):", sscd_corr3[0,1])

    fig = plt.figure()
    plot(fig,
         logit(modelderm_caffe_df['caffe_p']), 
         logit(modelderm_pytorch_df['pytorch_p']),
         "modelderm.pdf",
         min_=-15,
         max_=5)

    plot(fig,
         logit(scanoma_tflite_df['tflite_p']), 
         logit(scanoma_pytorch_df['pytorch_p']),
         "scanoma.pdf")

    plot(fig,
         logit(sscd_tflite_df['tflite_p']), 
         logit(sscd_pytorch_df['pytorch_p']),
         "sscd_warped.pdf")

    plot(fig,
         logit(sscd_tflite_df['tflite_p_cropped']), 
         logit(sscd_pytorch_df['pytorch_p']),
         "sscd_cropped.pdf")

    plot(fig,
         logit(sscd_tflite_df['tflite_p_cropped']), 
         logit(sscd_pytorch_df['pytorch_p_nn']),
         "sscd_cropped_nn.pdf")

if __name__ == "__main__":
    main()
