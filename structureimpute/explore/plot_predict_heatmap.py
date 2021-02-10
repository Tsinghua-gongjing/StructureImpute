from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
import sys, os
from nested_dict import nested_dict
import pandas as pd
import numpy as np
from pyfasta import Fasta
import os
import re
import torch
import time
from termcolor import colored
import util
import argparse

def plot_heatmap(predict_ls, label_ls, start_index=40, end_index=60, extend=10, motif_len=7, savefn_meta=None, cmap=None, facecolor=None, plot_heatmap=0, plot_error=0):
    region_mean_ls = []
    region_std_ls = []
    region_num_ls = []
    for predict,label in zip(predict_ls.split(':'), label_ls.split(':')):
        df = pd.read_csv(predict, header=None, sep=',')
        print(df.head())
        df = df.iloc[:,50-motif_len-extend:50+extend]

        if plot_heatmap:
            savefn = predict.replace('.txt', '.heatmap.pdf')
            fig,ax=plt.subplots(figsize=(10,20))
            g = sns.heatmap(df, xticklabels=False, yticklabels=False, vmax=1, vmin=0, cmap=cmap)
            g.set_facecolor(facecolor)
            plt.savefig(savefn)
            plt.close()
        
        region_mean = list(df.mean())
        region_mean_ls.append(region_mean)
        region_std_ls.append(list(df.std()))
        region_num_ls.append(df.shape[0])
        
    fig,ax=plt.subplots(figsize=(16,8))
    for n,(region_mean,label,region_num,region_std) in enumerate(zip(region_mean_ls, label_ls.split(':'), region_num_ls, region_std_ls)):
        if plot_error:
            print(label, region_std)
            plt.errorbar([i+{0:-0.1,1:0.1}[n] for i in range(len(region_mean))], y=region_mean, yerr=region_std, label="{}(n={})".format(label, region_num), marker='.', )
        else:
            ax.plot(region_mean, label="{}(n={})".format(label, region_num), marker='.', )
    ax.set_ylim(0,0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefn_meta)
    plt.close()
    
    return 
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot predict')
    
    parser.add_argument('--predict_ls', type=str, default='/home/gongjing/project/shape_imputation/exper/95_trainLossall_GmultiplyX_null0.1x10/prediction.LIN28_exceed.txt:/home/gongjing/project/shape_imputation/exper/95_trainLossall_GmultiplyX_null0.1x10/prediction.LIN28_ok.txt', help='Path to predict file')
    parser.add_argument('--label_ls', type=str, default='exceed:ok', help='Label of plot')
    parser.add_argument('--start_index', type=int, default=40, help='Plot start index of df')
    parser.add_argument('--end_index', type=int, default=60, help='Plot end index of df')
    parser.add_argument('--extend', type=int, default=10, help='Extend length of motif')
    parser.add_argument('--motif_len', type=int, default=7, help='Length of center motif')
    parser.add_argument('--plot_heatmap', type=int, default=1, help='Whether to plot(1) heatmap or not(0)')
    parser.add_argument('--plot_error', type=int, default=0, help='Whether to plot(1) std in meta or not(0)')
    parser.add_argument('--savefn_meta', type=str, default='/home/gongjing/project/shape_imputation/exper/95_trainLossall_GmultiplyX_null0.1x10/prediction.LIN28.pdf', help='Save file to meta shape profile')
    parser.add_argument('--cmap', type=str, default='RdYlGn_r', help='Heatmap color map')
    parser.add_argument('--facecolor', type=str, default='#CBCBCB', help='Heatmap background color (for NULL value)')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot predict', args)
    
    plot_heatmap(predict_ls=args.predict_ls, label_ls=args.label_ls, start_index=args.start_index, end_index=args.end_index, extend=args.extend, motif_len=args.motif_len, savefn_meta=args.savefn_meta, cmap=args.cmap, facecolor=args.facecolor, plot_heatmap=args.plot_heatmap, plot_error=args.plot_error)

if __name__ == '__main__':
    main()