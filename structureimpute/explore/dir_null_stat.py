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
import os
import re
import torch
import time
from termcolor import colored
import util
import argparse
import itertools

import data_shape_distribution

def read_null_pattern(null_stat):
    d = {}
    with open(null_stat, 'r') as STAT:
        for line in STAT:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            for pos in range(int(arr[4]), int(arr[5])):
                base_pos = int(arr[2]) + pos
                d[arr[0]+'|'+str(base_pos)] = 1
    return d

def null_stat_compare(null_stat1=None, null_stat2=None):
    if null_stat1 is None:
        null_stat1 = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.null_pattern.txt'
    if null_stat2 is None:
        null_stat2 = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_625178.null_pattern.txt'
    null_stat1_d = read_null_pattern(null_stat1)
    null_stat2_d = read_null_pattern(null_stat2)
    in1 = set(null_stat1_d.keys()) - set(null_stat2_d.keys())
    in12 = set(null_stat1_d.keys()) & set(null_stat2_d.keys())
    in2 = set(null_stat2_d.keys()) - set(null_stat1_d.keys())
    in1or2 = set(null_stat2_d.keys()) | set(null_stat1_d.keys())
    d = {}
    d['in1'] = len(in1)
    d['in2'] = len(in2)
    d['in12'] = len(in12)
    d['len1'] = len(null_stat1_d)
    d['len2'] = len(null_stat2_d)
    d['union'] = len(in1or2)
    print(null_stat1, null_stat2, d)
    return d

def dir_null_pos_overlap(d, fn_ls=None):
    if fn_ls is None:
        fn_ls = os.listdir(d)
        fn_ls = [i for i in fn_ls if '.txt' in i and '.pdf' not in i and 'null_pattern' in i and 'x10' not in i]
        
    stat_ls = set(['_'.join('.'.join(i.split('.')[-4:-2]).split('_')[0:-1]) for i in fn_ls])
    
    for i in list(stat_ls)[0:]:
        pair_d = nested_dict(2, list)
        fn_ls_sub = [f for f in fn_ls if i in f]
        # print(i)
        
        fn_pair = itertools.combinations(fn_ls_sub, 2)
        for fn1,fn2 in fn_pair:
            print(fn1,fn2)
            stat_d = null_stat_compare(null_stat1=d+'/'+fn1, null_stat2=d+'/'+fn2)
            fn1_label = fn1.replace('windowLen100.sliding100.', '').replace('.null_pattern.txt', '')
            fn2_label = fn2.replace('windowLen100.sliding100.', '').replace('.null_pattern.txt', '')
            pair_d[fn1_label+'\n({})'.format(stat_d['len1'])][fn2_label+'\n({})'.format(stat_d['len2'])] = int(stat_d['in12'])
            pair_d[fn2_label+'\n({})'.format(stat_d['len2'])][fn1_label+'\n({})'.format(stat_d['len1'])] = int(stat_d['in12'])
            
        pair_df = pd.DataFrame.from_dict(pair_d)
        print(pair_df)
        
        savefn = d+'/null_pos_compare.{}.txt'.format(i)
        pair_df.to_csv(savefn, header=True, index=True, sep='\t')
        
        savefn = savefn.replace('.txt', '.pdf')
        util.heatmap(pair_df, xlabel='sample1', ylabel='sample2', savefn=savefn, fmt='0.0f', fig_size_x=20, fig_size_y=20)
        

def fn_ls_null_pos_overlap(fn_ls=None, label_ls=None, savefn=None):
    pair_d = nested_dict(2, list)
    fn_label_dict = {i:j for i,j in zip(fn_ls, label_ls)}
        
    fn_pair = itertools.combinations(fn_ls, 2)
    for fn1,fn2 in fn_pair:
        print(fn1,fn2)
        stat_d = null_stat_compare(null_stat1=fn1, null_stat2=fn2)
        fn1_label = fn_label_dict[fn1]
        fn2_label = fn_label_dict[fn2]
        pair_d[fn1_label+'\n({})'.format(stat_d['len1'])][fn2_label+'\n({})'.format(stat_d['len2'])] = int(stat_d['in12'])
        pair_d[fn2_label+'\n({})'.format(stat_d['len2'])][fn1_label+'\n({})'.format(stat_d['len1'])] = int(stat_d['in12'])
            
    pair_df = pd.DataFrame.from_dict(pair_d)
    print(pair_df)
        
    pair_df.to_csv(savefn, header=True, index=True, sep='\t')
        
    savefn = savefn.replace('.txt', '.pdf')
    util.heatmap(pair_df, xlabel='sample1', ylabel='sample2', savefn=savefn, fmt='0.0f', fig_size_x=20, fig_size_y=20)
        
def dir_null_stat(d):
    fn_ls = os.listdir(d)
    fn_ls = [i for i in fn_ls if '.txt' in i and '.pdf' not in i and 'null_pattern' not in i and 'x10' not in i]
    df_stat_ls = []
    df_dict = {}
    for fn in fn_ls:
        sample = '|'.join(fn.split('.')[-3:-1])
        fn_path = d+'/'+fn
        savefn = fn_path.replace('.txt', '.pdf')
        df_stat,df = data_shape_distribution.null_count_dist(data=fn_path, col='7:8', normalize=True, savefn=savefn)
        df_stat.columns = [sample]
        df_stat_ls.append(df_stat)
        df_dict[sample] = df
        
    df_stat_all = pd.concat(df_stat_ls, axis=1)
    print(df_stat_all)
    
    fig,ax=plt.subplots(figsize=(30,40))
    df_stat_all_T = df_stat_all.T
    df_stat_all_T.sort_index(inplace=True)
    df_stat_all_T.plot(kind='barh', stacked=True, ax=ax)
    plt.tight_layout()
    plt.savefig(d+'/all_sample_null_count.pdf')
    plt.close()
        
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Calc NULL/nonNULL value distribution in train/validation set')
    
    parser.add_argument('--d', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling', help='Path to dir')
    
    # get args
    args = parser.parse_args()
    # dir_null_stat(d=args.d)
    
    # null_stat_compare()
    # dir_null_pos_overlap(d=args.d)
    
    pct3_maxL20_seed_ls = ['1113', '12315', '1234', '19491001', '19930426', '2019', '400100', '42', '5678', '9999']
    pct3_maxL20_fn_ls = ['/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.S{}.null_pattern.txt'.format(i) for i in pct3_maxL20_seed_ls]
    pct3_maxL20_label_ls = ['seed{}'.format(i) for i in pct3_maxL20_seed_ls]
    savefn = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/null_pos_compare.pct3_maxL20.txt'
    # fn_ls_null_pos_overlap(fn_ls=pct3_maxL20_fn_ls, label_ls=pct3_maxL20_label_ls, savefn=savefn)
    
    sampling_120M_seed_ls = ['1234', '40', '9988', '17181790', '81910', '625178', '1', '7829999', '9029102', '918029109']
    sampling_120M_fn_ls = ['/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_25_{}.null_pattern.txt'.format(i) for i in sampling_120M_seed_ls]
    sampling_120M_label_ls = ['120M_seed{}'.format(i) for i in sampling_120M_seed_ls]
    savefn = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/new.null_pos_compare.pct3_maxL20.vs_50M.txt'
    fn_ls_null_pos_overlap(fn_ls=sampling_120M_fn_ls, label_ls=sampling_120M_label_ls, savefn=savefn)
    
    sampling_120M_10 = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_60_1234.null_pattern.x10.chrom1.null_sequential.txt'
    pct3_maxL20_10 = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.x10.null_pattern.txt'
    savefn = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/null_pos_compare.pct3_maxL20.x10.vs_120Mchrom1x10.txt'
    # fn_ls_null_pos_overlap(fn_ls=[pct3_maxL20_10,sampling_120M_10], label_ls=['pct30%x10','120Mx10'], savefn=savefn)
    
    sampling_120M_1 = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_60_1234.null_pattern.txt'
    pct3_maxL20_1 = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.S1234.null_pattern.txt'
    savefn = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/null_pos_compare.pct3_maxL20.vs_120Mchrom1x10.txt'
    # fn_ls_null_pos_overlap(fn_ls=[pct3_maxL20_1,sampling_120M_1], label_ls=['pct30%x1','120Mx1'], savefn=savefn)
    
    sampling_120M_shuffle_seed_ls = ['1234', '40', '9988', '17181790', '81910', '625178', '1', '7829999', '9029102', '918029109']
    sampling_120M_shuffle_fn_ls = ['/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_60_1234.null_pattern.s{}.chrom0.null_pattern.txt'.format(i) for i in sampling_120M_shuffle_seed_ls]
    sampling_120M_shuffle_label_ls = ['120M_shuffle_seed{}'.format(i) for i in sampling_120M_shuffle_seed_ls]
    savefn = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/null_pos_compare.120M.chrom0.shuffle_compare.txt'
    # fn_ls_null_pos_overlap(fn_ls=sampling_120M_shuffle_fn_ls, label_ls=sampling_120M_shuffle_label_ls, savefn=savefn)
    
    
    
if __name__ == '__main__':
    main()