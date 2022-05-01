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
import os, subprocess
import re
import torch
import time
from termcolor import colored
import util
import argparse

def plot_shape_tx_null_pct(out1=None, out2=None, out1_label='True', out2_label='Predict', savefn=None, species_fa=None, species='human'):
    out_dict1 = util.read_icshape_out(out1)
    out_dict2 = util.read_icshape_out(out2)
    tx_common = set(out_dict1.keys()) & set(out_dict2.keys())
    null_pct1_ls = []
    null_pct2_ls = []
    for tx in tx_common:
        null_pct1 = (out_dict1[tx]['reactivity_ls'].count('NULL')+out_dict1[tx]['reactivity_ls'].count('-1.0')+out_dict1[tx]['reactivity_ls'].count('-1')) / float(out_dict1[tx]['length'])
        null_pct2 = (out_dict2[tx]['reactivity_ls'].count('NULL')+out_dict2[tx]['reactivity_ls'].count('-1.0')+out_dict1[tx]['reactivity_ls'].count('-1')) / float(out_dict2[tx]['length'])
        null_pct1_ls.append(null_pct1)
        null_pct2_ls.append(null_pct2)
    print('{}: n={}'.format(out1, len(out_dict1)))
    print('{}: n={}'.format(out2, len(out_dict2)))
    print('common tx: n={}'.format(len(tx_common)))
    
    fa_dict = util.read_fa(fa=species_fa, species=species, pureID=1)
    stat1 = util.shape_dict_stat(out_dict1, fa_dict, None, RNA_type=None, trim5Len=5, trim3Len=30)
    stat2 = util.shape_dict_stat(out_dict2, fa_dict, None, RNA_type=None, trim5Len=5, trim3Len=30)
    print(pd.DataFrame.from_dict(stat1,orient='index'), pd.DataFrame.from_dict(stat2, orient='index'))
    
    df = pd.DataFrame.from_dict({out1_label:null_pct1_ls, out2_label:null_pct2_ls})
    print(df.head())
    fig,ax=plt.subplots(figsize=(6,6))
    sns.scatterplot(x=out1_label, y=out2_label, data=df, ax=ax, s=10)
    plt.xlabel('{} (null_pct: {:.2f})'.format(out1_label, stat1['total_bases(NULL_pct)']))
    plt.ylabel('{} (null_pct: {:.2f})'.format(out2_label, stat2['total_bases(NULL_pct)']))
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    return stat1,stat2
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot null pct scatter of common tx between two icshape.out')
    
    parser.add_argument('--icshape1', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out', help='icSHAPE out file1')
    parser.add_argument('--icshape2', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.allfragment.0.5+exceed0.5.txt2.predict.out', help='icSHAPE out file2')
    parser.add_argument('--out1_label', type=str, default='True', help='icSHAPE out file1 label')
    parser.add_argument('--out2_label', type=str, default='Predict', help='icSHAPE out file2 label')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.allfragment.0.5+exceed0.5.txt2.predict.out.scatter.pdf', help='Save plot file')
    parser.add_argument('--species_fa', type=str, default=None, help='Species .fa reference file')
    parser.add_argument('--species', type=str, default='human', help='Species')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot null pct scatter of common tx between two icshape.out', args)
    plot_shape_tx_null_pct(out1=args.icshape1, out2=args.icshape2, out1_label=args.out1_label, out2_label=args.out2_label, savefn=args.savefn, species_fa=args.species_fa, species=args.species)
    

if __name__ == '__main__':
    main()