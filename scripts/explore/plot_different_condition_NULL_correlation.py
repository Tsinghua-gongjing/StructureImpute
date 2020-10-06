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

def plot_corr_bar(condition_ls, correlation_ls, savefn):
    condition_ls = condition_ls.split(':')
    correlation_ls = list(map(float, correlation_ls.split(':')))
    
    df = pd.DataFrame.from_dict({'Condition':condition_ls, 'Correlation':correlation_ls})
    
    fig,ax=plt.subplots()
    sns.barplot(x='Condition', y='Correlation', data=df, order=condition_ls)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot correlation bar of multiple condition')
    
    # parser.add_argument('--condition_ls', type=str, default='seq+shape:seq:shape', help='Condition list')
    # parser.add_argument('--correlation_ls', type=str, default='0.78:0.305:0.343', help='Correlation list')
    # parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/condition_compare_correlation.pdf', help='Path to plot file')
    
    parser.add_argument('--condition_ls', type=str, default='HEK293(wc, vivo):HEK293(wc, vitro):HEK293(ch, vivo):HEK293(np, vivo):HEK293(cy, vivo):mES(wc, vivo)', help='Condition list')
    parser.add_argument('--correlation_ls', type=str, default='0.78:0.702:0.704:0.761:0.723:0.675', help='Correlation list')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/condition_compare_correlation.vivo_vitro_WcChNpCy.pdf', help='Path to plot file')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot correlation bar of multiple condition', args)
    
    plot_corr_bar(condition_ls=args.condition_ls, correlation_ls=args.correlation_ls, savefn=args.savefn)

if __name__ == '__main__':
    main()
    
'''
python plot_different_condition_NULL_correlation.py --condition_ls "seq+shape:seq:shape:shaker" --correlation_ls '0.60:0.347:0.15:0.274' --savefn /home/gongjing/project/shape_imputation/results/method_compare_correlation.vs_seq_shape_shaker.pdf
'''