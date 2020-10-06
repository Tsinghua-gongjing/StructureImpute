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

def reactivity_dist(data, col='7:8', normalize=True, savefn=None):
    reactivity_dict = nested_dict(1, list)
    col_ls = list(map(int, col.split(':')))
    with open(data, 'r') as FILE:
        for line in FILE:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            shape1 = arr[col_ls[0]].split(',')
            shape2 = arr[col_ls[1]].split(',')
            for i,j in zip(shape1, shape2):
                if float(i) == -1:
                    reactivity_dict['null(true)'].append(float(j))
                else:
                    reactivity_dict['nonnull(true)'].append(float(j))
    reactivity_stat_dict = nested_dict(2, list)
    for i,j in reactivity_dict.items():
        df = pd.DataFrame({'reactivity':j})
        print('total base', i, len(j))
        df['bins'] = pd.cut(df['reactivity'], bins=10)
        count_dict = dict(df['bins'].astype(str).value_counts(normalize=normalize))
        for m,n in count_dict.items():
            print(m,n)
            reactivity_stat_dict[i][m] = n
        # print(i, df['bins'].value_counts())
    reactivity_stat_df = pd.DataFrame.from_dict(reactivity_stat_dict, orient='columns')
    reactivity_stat_df.rename(columns={'null(true)':'null(true)\n(n={})'.format(len(reactivity_dict['null(true)'])), 'nonnull(true)':'nonnull(true)\n(n={})'.format(len(reactivity_dict['nonnull(true)']))}, inplace=True)
    print(reactivity_stat_df)
    
    fig,ax=plt.subplots(figsize=(12, 6))
    reactivity_stat_df.T.plot(kind='barh', stacked=True, ax=ax)
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.savefig(savefn, bbox_inches='tight')
    plt.close()
    
    return reactivity_dict.to_dict()

def null_count_dist(data, col='7:8', normalize=True, savefn=None):
    col_ls = list(map(int, col.split(':')))
    savefn = savefn.replace('.pdf', '.null_pattern.txt')
    SAVEFN = open(savefn, 'w')
    with open(data, 'r') as FILE:
        for line in FILE:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            reactivity_ls = arr[col_ls[0]].replace('-1', 'NULL').split(',')
            pos_ls = util.reactivity_ls_null_loc(reactivity_ls, start=0, trim5Len=0)
            for pos in pos_ls:
                SAVEFN.write('\t'.join(map(str, arr[0:4]+[pos[0], pos[1], pos[1]-pos[0]]))+'\n')
    SAVEFN.close()
    
    df = pd.read_csv(savefn, sep='\t', header=None)
    df.columns = ['tx', 'length', 'fragment_start', 'fragment_end', 'start', 'end', 'len']
    df['len2'] = [10 if float(i)>=10 else i for i in df['len'] ]
    
    fig,ax=plt.subplots()
    df_stat = pd.DataFrame(df.groupby('len2').count()['tx'])
    print(df_stat)
    sns.barplot(data=df_stat.T,ax=ax)
    plt.tight_layout()
    plt.savefig(savefn.replace('.txt', '.pdf'))
    plt.close()
    
    return df_stat,df
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Calc NULL/nonNULL value distribution in train/validation set')
    
    parser.add_argument('--data', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/c200T2/w100s100.train_null0.1.txt', help='Path to fragment file')
    parser.add_argument('--col', type=str, default='7:8', help='Columns to calc')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/c200T2/w100s100.train_null0.1.stat.pdf', help='Path to save stat file')
    parser.add_argument('--dist_type', type=str, default='null_pattern', help='Type of dist')
    
    # get args
    args = parser.parse_args()
    util.print_args('Calc NULL/nonNULL value distribution in train/validation set', args)
    
    if 'null_pattern' in args.dist_type:
        null_count_dist(data=args.data, col=args.col, savefn=(args.data).replace('.txt', '.pdf'))
    if 'shape_value' in args.dist_type:
        reactivity_dist(data=args.data, col=args.col, savefn=args.savefn)

if __name__ == '__main__':
    main()
    
'''
python data_shape_distribution.py --data /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.x10.txt
'''