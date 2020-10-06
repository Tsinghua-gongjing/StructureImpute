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

def plot_sample(f=None, savefn=None, cmap=None, facecolor=None, col=None, null_type=None):
    df = pd.read_csv(f, header=None, sep='\t')
    if null_type == 'null_count':
        df['n(NULL)'] = [i.split(',').count('NULL') for i in df[col]]
    if null_type == 'null_1st_pos':
        df['n(NULL)'] = [i.split(',').index('-1') for i in df[col]]
    df.sort_values(by='n(NULL)', inplace=True)
    print(df['n(NULL)'].value_counts())
    print(df.head())

    # df = df[df['n(NULL)']<=26]
    df_3 = df[col].str.split(",",expand=True,)
    df_3.replace('NULL', np.nan, inplace=True)
    df_3.replace('-1', np.nan, inplace=True)
    df_3 = df_3.applymap(lambda x:float(x))
    print(df_3.shape)

    fig,ax=plt.subplots(figsize=(10,20))
    g = sns.heatmap(df_3.head(100), xticklabels=False, yticklabels=False, cmap=cmap, linewidths=0)
    g.set_facecolor(facecolor)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf', '.h100.pdf'))
    plt.close()

    fig,ax=plt.subplots(figsize=(10,20))
    g = sns.heatmap(df_3.head(1000), xticklabels=False, yticklabels=False, cmap=cmap, linewidths=0)
    g.set_facecolor(facecolor)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf', '.h1000.pdf'))
    plt.close()

    fig,ax=plt.subplots(figsize=(10,20))
    g = sns.heatmap(df_3, xticklabels=False, yticklabels=False, cmap=cmap, linewidths=0)
    g.set_facecolor(facecolor)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf', '.all.pdf'))
    plt.close()
  
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot sample heatmap')
    parser.add_argument('--f', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.S1234.txt', help='Path to sample file')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/hek_train_pct0.3L20.pdf', help='Save file to meta shape profile')
    parser.add_argument('--cmap', type=str, default='summer', help='Heatmap color map')
    parser.add_argument('--facecolor', type=str, default='black', help='Heatmap background color (for NULL value)')
    parser.add_argument('--col', type=int, default=7, help='Value col index')
    parser.add_argument('--null_type', type=str, default='null_1st_pos', help='null_1st_pos(for validation sample)|null_count(for m6A/RBP/start/stopcodon)')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot sample heatmap', args)
    plot_sample(f=args.f, savefn=args.savefn, cmap=args.cmap, facecolor=args.facecolor, col=args.col, null_type=args.null_type)
                
if __name__ == '__main__':
    main()
    
"""
# for validation sample
python plot_sample_heatmap.py --cmap RdYlGn_r --facecolor "#CBCBCB"

# FXR2 
python plot_sample_heatmap.py --f /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.FXR.txt.shape --savefn /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.FXR.txt.shape.0317.pdf --cmap RdYlGn_r --facecolor "#CBCBCB" --col 14 --null_type null_count
python plot_sample_heatmap.py --f /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt.shape --savefn /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt.shape.0317.pdf --cmap RdYlGn_r --facecolor "#CBCBCB" --col 14 --null_type null_count

# for m6A
python plot_sample_heatmap.py --f /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_null.e0.bed.shape --savefn /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_null.e0.bed.shape.RdYLGn_r.pdf --cmap RdYlGn_r --facecolor "#CBCBCB" --col 3 --null_type null_count

python plot_sample_heatmap.py --f /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_valid.e0.bed.shape --savefn /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_valid.e0.bed.shape.RdYLGn_r.pdf --cmap RdYlGn_r --facecolor "#CBCBCB" --col 3 --null_type null_count
"""