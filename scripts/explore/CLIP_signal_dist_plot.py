from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
import sys, os,subprocess
from nested_dict import nested_dict
import pandas as pd
import numpy as np
from pyfasta import Fasta
import os,subprocess
import re
import torch
import time
from termcolor import colored
import util
import argparse
import itertools
from scipy import stats

def read_clip_signal(bed=None):
    if bed is None:
        bed = '/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.bed'
    region_signal_dict = nested_dict(1, float)
    with open(bed, 'r') as BED:
        for line in BED:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            region = '{}:{}-{}'.format(arr[-3].split('.')[0], arr[-2], arr[-1])
            score = arr[4].split('|')[-1]
            region_signal_dict[region] = float(score)
    # print(region_signal_dict)
    return region_signal_dict.to_dict()

def plot_score_dist(signal_bed, plot_bed_ls, plot_bed_label_ls, savefn):
    region_signal_dict = read_clip_signal(bed=signal_bed)
    plot_bed_ls = plot_bed_ls.split(',')
    plot_bed_label_ls = plot_bed_label_ls.split(',')
    bed_signal_ls_ls = []
    for plot_bed, plot_bed_label in zip(plot_bed_ls, plot_bed_label_ls):
        bed_signal_ls = []
        with open(plot_bed, 'r') as BED:
            for line in BED:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                arr = line.split('\t')
                # region = '{}:{}-{}'.format(arr[0], arr[1], arr[2])
                region = arr[6]
                if region in region_signal_dict:
                    signal = region_signal_dict[region]
                else:
                    signal = np.nan
                bed_signal_ls.append(signal)
        bed_signal_ls_ls.append(bed_signal_ls)
        
    # print(bed_signal_ls_ls)
    df_bed_ls = []
    for i,j in zip(bed_signal_ls_ls, plot_bed_label_ls):
        df_bed = pd.DataFrame.from_dict({'signal':i, 'label':'{}(n={})'.format(j, len(i))})
        df_bed_ls.append(df_bed)
    df_bed_all = pd.concat(df_bed_ls, axis=0)
    print(df_bed_all)
    
    fig,ax=plt.subplots(figsize=(3*len(plot_bed_label_ls), 6))
    sns.boxplot(x='label',y='signal', data=df_bed_all)
    r,p = stats.ttest_ind(bed_signal_ls_ls[0],bed_signal_ls_ls[1])
    p2=stats.ks_2samp(bed_signal_ls_ls[0],bed_signal_ls_ls[1])[1]
    print(r,p,p2)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
        
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot a list of .bed region signal dist')
    
    parser.add_argument('--signal_bed', type=str, default='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.bed', help='Bed file of CLIP signal')
    parser.add_argument('--plot_bed_ls', type=str, default='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.FXR.txt.shape,/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt.shape', help='List of bed file regions to check')
    parser.add_argument('--plot_bed_label_ls', type=str, default='Valid,NULL', help='Label for a list of bed file regions to check')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/CLIPDB20123_FXR2_HEK293_trx.signal_dist.pdf', help='savefn')
    
    # get args
    args = parser.parse_args()
    plot_score_dist(signal_bed=args.signal_bed, plot_bed_ls=args.plot_bed_ls, plot_bed_label_ls=args.plot_bed_label_ls, savefn=args.savefn)
        
if __name__ == '__main__':
    main()
    
'''
python CLIP_signal_dist_plot.py --signal_bed /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20161_IGF2BP1_HEK293_trx.bed --plot_bed_ls /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20161_IGF2BP1_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.IGF2BP1_11.txt.shape,/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20161_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.IGF2BP1_11.txt.shape --savefn /home/gongjing/project/shape_imputation/results/CLIPDB20161_IGF2BP1_win11_HEK293_trx.signal_dist.pdf

python CLIP_signal_dist_plot.py --signal_bed /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.bed --plot_bed_ls /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.IGF2BP1_11.txt.shape,/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.IGF2BP1_11.txt.shape --savefn /home/gongjing/project/shape_imputation/results/CLIPDB20162_IGF2BP1_win11_HEK293_trx.signal_dist.pdf
'''