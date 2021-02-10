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

def parse_align(train_fa, validation_fa, blastn_output, savefn):
    train_fa_dict = Fasta(train_fa)
    validation_fa_dict = Fasta(validation_fa)
    seq_similarity_dict = nested_dict(2, list)
    for i in list(validation_fa_dict.keys()):
        for j in list(train_fa_dict.keys()):
            seq_similarity_dict[i][j] = np.nan
    with open(blastn_output, 'r') as OUT:
        for line in OUT:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            seq_similarity_dict[arr[0]][arr[1]] = -np.log10(float(arr[10]))
    seq_similarity_df = pd.DataFrame.from_dict(seq_similarity_dict, orient='index')
    
    fig,ax=plt.subplots(figsize=(12,30))
    sns.heatmap(seq_similarity_df.T.head(1000),xticklabels=False, yticklabels=False, cmap="YlGnBu")
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    return seq_similarity_df

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot heatmap of pair wise heatmap for two sequence set alignments')
    
    parser.add_argument('--train_fa', type=str, default='/home/gongjing/project/shape_imputation/data/seq_similarity/windowLen100.sliding100.train.fa', help='Path to reference(db) file, here train file')
    parser.add_argument('--validation_fa', type=str, default='/home/gongjing/project/shape_imputation/data/seq_similarity/windowLen100.sliding100.validation.fa', help='Path to query file, here validation file')
    parser.add_argument('--blastn_output', type=str, default='/home/gongjing/project/shape_imputation/data/seq_similarity/train_validation_seq_align.txt', help='Path to alignmnet file')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/data/seq_similarity/train_validation_seq_align.heatmap.pdf', help='Path to alignmnet heatmap file')
    
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot heatmap of pair wise heatmap for two sequence set alignments', args)
    
    parse_align(train_fa=args.train_fa, validation_fa=args.validation_fa, blastn_output=args.blastn_output, savefn=args.savefn)

if __name__ == '__main__':
    main()