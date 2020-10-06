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

def plot_corr_bar(condition_ls, SHAPEImpute_ls, ShaKer_ls, savefn):
    condition_ls = condition_ls.split(':')
    SHAPEImpute_ls = list(map(float, SHAPEImpute_ls.split(':')))
    ShaKer_ls = list(map(float, ShaKer_ls.split(':')))
    
    df = pd.DataFrame.from_dict({'Condition':condition_ls+condition_ls, 'Correlation':SHAPEImpute_ls+ShaKer_ls, 'Method':['SHAPEImpute']*len(SHAPEImpute_ls) + ['ShaKer']*len(ShaKer_ls)})
    print(df)
    
    fig,ax=plt.subplots()
    sns.barplot(x='Condition', y='Correlation', hue='Method', data=df, order=condition_ls)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot correlation bar of multiple condition')
    
    parser.add_argument('--condition_ls', type=str, default='hek_wc_vivo:hek_wc_vitro:hek_ch_vivo:hek_np_vivo:hek_cy_vivo:mes_wc_vivo', help='Condition list')
    parser.add_argument('--SHAPEImpute_ls', type=str, default='0.766:0.715:0.699:0.755:0.710:0.666', help='Correlation list')
    parser.add_argument('--ShaKer_ls', type=str, default='0.274:0.256:0.226:0.228:0.242:0.264', help='Correlation list')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/condition_compare_correlation.shapeimpute_vs_shaker_multiple_set.pdf', help='Path to plot file')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot correlation bar of multiple condition', args)
    
    plot_corr_bar(condition_ls=args.condition_ls, SHAPEImpute_ls=args.SHAPEImpute_ls, ShaKer_ls=args.ShaKer_ls, savefn=args.savefn)

if __name__ == '__main__':
    main()
    
'''
python plot_SHAPEImpute_vs_Shaker_on_multiple_set.py --SHAPEImpute_ls '0.758:0.687:0.714:0.750:0.690:0.662' --ShaKer_ls '0.274:0.256:0.226:0.228:0.242:0.264' --savefn /home/gongjing/project/shape_imputation/results/condition_compare_correlation.shapeimpute_vs_shaker_multiple_set.c80.null0.3.pdf
'''