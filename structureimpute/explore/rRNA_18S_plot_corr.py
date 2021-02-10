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
    
    parser.add_argument('--condition_ls', type=str, default='0.1:0.2:0.3:0.4:0.5)', help='Condition list')
    parser.add_argument('--correlation_ls', type=str, default='0.940:0.915:0.885:0.810:0.755', help='Correlation list')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/condition_compare_correlation.18S.pdf', help='Path to plot file')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot correlation bar of multiple condition', args)
    
    plot_corr_bar(condition_ls=args.condition_ls, correlation_ls=args.correlation_ls, savefn=args.savefn)

if __name__ == '__main__':
    main()
    
'''
python rRNA_18S_plot_corr.py --condition_ls "true:0.1:0.2:0.3:0.4:0.5" --correlation_ls "0.715:0.720:0.707:0.718:0.701:0.701" --savefn /home/gongjing/project/shape_imputation/results/condition_compare_correlation.18S.validbase.pdf

python rRNA_18S_plot_corr.py --condition_ls "0.1:0.2:0.3:0.4:0.5" --correlation_ls "0.963:0.941:0.905:0.832:0.774" --savefn /home/gongjing/project/shape_imputation/results/condition_compare_correlation.18S.c94.pdf
'''