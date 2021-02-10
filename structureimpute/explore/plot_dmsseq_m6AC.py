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

from scipy import stats

def plot_m6AC(shape_ls, label_ls, savefn):
    shape_ls = shape_ls.split(':')
    label_ls = label_ls.split(':')
    
    value_dict = nested_dict(2, list)
    
    df = pd.read_csv(shape_ls[0], header=None, sep='\t')
    columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df.columns = columns
    print(df.head())
    
    for i in df['fragment_shape']:
        v = i.split(',')
        value_dict[label_ls[0]]['A'].append(float(v[49]))
        value_dict[label_ls[0]]['C'].append(float(v[50]))
        
    df_null = pd.read_csv(shape_ls[1], header=None, sep='\t')
    print(df_null.head())
    for i in df_null[0]:
        v = i.split(',')
        value_dict[label_ls[1]]['A'].append(float(v[49]))
        value_dict[label_ls[1]]['C'].append(float(v[50]))
        
    value_ls = [] #[value_dict[label_ls[0]]['A'], value_dict[label_ls[0]]['C'], value_dict[label_ls[1]]['A'], value_dict[label_ls[1]]['C']]
    value_label_ls = [] 
    for i,j in value_dict.items():
        for m,n in j.items():
            label = i+' '+m
            for value in n:
                value_ls.append(value)
                value_label_ls.append(label)
    value_df = pd.DataFrame.from_dict({'value':value_ls, 'label':value_label_ls})
    print(value_df.head())
    
    fig,ax=plt.subplots(figsize=(8,8))
    sns.boxplot(x='label', y='value', data=value_df)
    r1,p1 = stats.ttest_ind(value_dict[label_ls[0]]['A'], value_dict[label_ls[1]]['A'])
    r2,p2 = stats.ttest_ind(value_dict[label_ls[0]]['C'], value_dict[label_ls[1]]['C'])
    print(p1, p2)
    title = 'p1: {:.3f}, p2: {:.3f}'.format(p1,p2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot dms-seq m6AC score dist')
    
    parser.add_argument('--shape_ls', type=str, default='/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_valid.bed.shape100.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.dmsseq_k562_vivo_m6A_null.txt', help='List of shape file')
    parser.add_argument('--label_ls', type=str, default='valid:null_predict', help='Label list')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/dmsseq_m6AC_valid_vs_nullpredict.pdf', help='Savefn pdf')
    
    args = parser.parse_args()
    util.print_args('Plot dms-seq m6AC score dist', args)
    
    plot_m6AC(shape_ls=args.shape_ls, label_ls=args.label_ls, savefn=args.savefn)
    
if __name__ == '__main__':
    main()
    
'''
python plot_dmsseq_m6AC.py --shape_ls /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_valid.bed.shape100.txt:/home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.dmsseq_k562_vivo_m6A_null.txt --savefn /home/gongjing/project/shape_imputation/results/dmsseq_m6AC_valid_vs_nullpredict.d10.pdf

python plot_dmsseq_m6AC.py --shape_ls /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_valid.bed.shape100.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.dmsseq_k562_vivo_m6A_nullNullLessthan75.txt --savefn /home/gongjing/project/shape_imputation/results/dmsseq_m6AC_valid_vs_nullpredict.d6NullLessthan75.pdf

python plot_dmsseq_m6AC.py --shape_ls /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_valid.bed.shape100.NullLessthan75.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.dmsseq_k562_vivo_m6A_nullNullLessthan75.txt --savefn /home/gongjing/project/shape_imputation/results/dmsseq_m6AC_validNullLessthan75_vs_nullpredict.d6NullLessthan75.pdf

python plot_dmsseq_m6AC.py --shape_ls /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1fibroblast.tx_has_shape_base_valid.bed.shape100.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.dmsseq_fibroblast_vivo_m6A_null.txt --savefn /home/gongjing/project/shape_imputation/results/dmsseq_m6AC_valid_vs_nullpredict.d6.e1fibroblast.pdf
'''
    