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
import random

from matplotlib import cm

def plot_bar(seq=None, shape_ls=None, colormap='RdYlGn_r', savefn=None):
    if shape_ls is None:
        shape_ls = [random.random() for i in range(len(seq))]
    shape_ls = np.array(shape_ls)
    if colormap == 'RdYlGn_r':
        colors = cm.RdYlGn_r(shape_ls, )
        plot = plt.scatter(shape_ls, shape_ls, c = shape_ls, cmap = colormap)
        plt.clf()
        plt.colorbar(plot)
        
    plt.bar(range(len(shape_ls)), shape_ls, color = colors)
    plt.xticks(range(len(seq)), list(seq))
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot a list of .bed region signal dist')
    
    parser.add_argument('--seq', type=str, default='AUCGAAGCGCCCAGCAA', help='Sequence of the shape')
    parser.add_argument('--shape_ls', type=str, help='Shape value')
    parser.add_argument('--colormap', type=str, default='RdYlGn_r', help='colormap#choose from: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/scheme_shape_bar.pdf', help='savefn')
    
    # get args
    args = parser.parse_args()
    plot_bar(seq=args.seq, shape_ls=args.shape_ls, colormap=args.colormap, savefn=args.savefn)
        
if __name__ == '__main__':
    main()