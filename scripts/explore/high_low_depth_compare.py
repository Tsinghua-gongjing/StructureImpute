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
from scipy import stats

def fragment_for_test(fragment_high=None, fragment_low=None):
    if fragment_high is None:
        fragment_high = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.txt'
    if fragment_low is None:
        fragment_low = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_30/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.txt'
    df_high = pd.read_csv(fragment_high, sep='\t', header=None)
    df_low = pd.read_csv(fragment_low, sep='\t', header=None)
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
    df_high.columns = cols
    df_low.columns = cols
    df_high_low = pd.merge(df_high, df_low, how='inner', on=['tx', 'start', 'end'])
    print(df_high_low.shape, df_high_low.head())
    
    df_high_low_select = df_high_low[(df_high_low['null_pct_x']==0)&(df_high_low['null_pct_y']<=0.1)&(df_high_low['null_pct_y']>0)]
    print(df_high_low_select.shape)
    
    savefn = fragment_low.replace('.txt', '.valid_inhigh.txt')
    df_high_low_select['fragment_shape_y'] = [i.replace('NULL','-1') for i in df_high_low_select['fragment_shape_y']]
    cols_select = ['tx', 'length_y', 'start', 'end', 'mean_reactivity_y', 'null_pct_y','seq_y','fragment_shape_y', 'fragment_shape_x']
    df_high_low_select[cols_select].to_csv(savefn, header=False, index=False, sep='\t')
    
    df_high_low_select = df_high_low[(df_high_low['null_pct_x']==0)&(df_high_low['null_pct_y']==0)]
    print(df_high_low_select.shape)
    
    savefn = fragment_low.replace('.txt', '.valid_both.txt')
    df_high_low_select['fragment_shape_y'] = [i.replace('NULL','-1') for i in df_high_low_select['fragment_shape_y']]
    cols_select = ['tx', 'length_y', 'start', 'end', 'mean_reactivity_y', 'null_pct_y','seq_y','fragment_shape_y', 'fragment_shape_x']
    df_high_low_select[cols_select].to_csv(savefn, header=False, index=False, sep='\t')

def fragment_compare_corr(fragment_common=None, savefn=None):
    if fragment_common is None:
        fragment_common = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.txt'
    df = pd.read_csv(fragment_common, header=None, sep='\t')
    
    m1_ls,m2_ls,r_ls,p_ls=[],[],[],[]
    for i,j in zip(df[7],df[8]):
        shape1 = list(map(float,i.split(',')))
        shape2 = list(map(float,j.split(',')))
        r,p = stats.pearsonr(shape1, shape2)
        r_ls.append(r)
        p_ls.append(p)
        m1_ls.append(np.mean(shape1))
        m2_ls.append(np.mean(shape2))
    df['r'] = r_ls
    df['p'] = p_ls
    df['m1'] = m1_ls
    df['m2'] = m2_ls
    df['-log10(p)'] = -np.log10(df['p'])

    fig,ax=plt.subplots(figsize=(10,10))
    g = sns.jointplot(x='m1',y='m2',data=df,kind='kde', xlim=(0.0,0.5), ylim=(0.0,0.5), height=8, ratio=5)
    sns.regplot(df['m1'],df['m2'], scatter=False, ax=g.ax_joint)

    r,p = stats.pearsonr(df['m1'],df['m2'])
    s = 'R = {:.2f}\nP = {:.2e}\nN = {}'.format(r,p,df.shape[0])
    g.ax_joint.text(0.05, 0.9, s, ha='left', va='top', size=20, transform=g.ax_joint.transAxes)
    
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    fig,ax=plt.subplots(figsize=(10,6))
    ax.hist(df['r'], bins=100)
    ax.set_xlabel('Pearson correlation coefficient of fragment')
    ax.set_ylabel('# of fragment')
    plt.axvline(x=np.mean(df['r']), ymin=0, ymax=1, hold=None, ls='--', color='red')
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf', '.hist.pdf'))
    plt.close()
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Retain common fragments in two validation file with different depth')
    parser.add_argument('--fragment_high', type=str, help='Path to fragment file with depth1')
    parser.add_argument('--fragment_low', type=str, help='Path to fragment file with depth2')
    parser.add_argument('--fragment_common', type=str, help='Path to fragment file with depth2')
    parser.add_argument('--savefn', type=str, help='Path to save plot file')
    parser.add_argument('--process_type', type=str, default='generate_common_fragment|corr_of_common_fragment', help='process')
    
    # get args
    args = parser.parse_args()
    util.print_args('Retain common fragments in two validation file with different depth', args)
    
    if args.process_type == 'generate_common_fragment':
        fragment_for_test(fragment_high=args.fragment_high, fragment_low=args.fragment_low)
    if args.process_type == 'corr_of_common_fragment':
        fragment_compare_corr(fragment_common=args.fragment_common, savefn=args.savefn)

if __name__ == '__main__':
    main()
    
'''
python high_low_depth_compare.py --process_type corr_of_common_fragment --fragment_common /home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.txt --savefn /home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.corr.pdf
'''