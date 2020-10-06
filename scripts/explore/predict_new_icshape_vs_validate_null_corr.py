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
from scipy import stats
import argparse

def read_validation_null_true(validation):
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validate = pd.read_csv(validation, header=None, sep='\t')
    df_validate.columns = cols
    
    validation_dict = nested_dict(2, list)
    
    for v,t,s,e,tx in zip(df_validate['fragment_shape'], df_validate['fragment_shape(true)'], df_validate['start'], df_validate['end'], df_validate['tx']):
        v_ls = v.split(',')
        t_ls = t.split(',')
        for n,i in enumerate(range(s,e)):
            if v_ls[n] in ['NULL','-1','-1.0']:
                pos = s + n
                validation_dict[tx][pos] = t_ls[n]
    return validation_dict.to_dict()

def compare_two_dict(out_dict, validation_dict, savefn):
    SAVEFN = open(savefn, 'w')
    true_ls = []
    predict_ls = []
    for i in validation_dict:
        if i not in out_dict: continue
        for j in validation_dict[i]:
            SAVEFN.write('\t'.join(map(str, [i, j, out_dict[i]['reactivity_ls'][j], validation_dict[i][j]]))+'\n')
            true_ls.append(float(out_dict[i]['reactivity_ls'][j]))
            predict_ls.append(float(validation_dict[i][j]))
    p=stats.pearsonr(true_ls,predict_ls)[0]
    print('corr:',p, 'base num:', len(true_ls))
    true_cal_ls,predict_cal_ls = [],[]
    for i,j in zip(true_ls, predict_ls):
        if i<0 or j<0: continue
        true_cal_ls.append(i)
        predict_cal_ls.append(j)
    p=stats.pearsonr(true_cal_ls,predict_cal_ls)[0]
    print('corr:',p, 'base num:', len(true_cal_ls))
    
    return p,len(true_ls),len(true_cal_ls)

def plot_dir_null_corr(args):
    validation_dict = read_validation_null_true(args.validation)
    
    fns = os.listdir(args.predict_dir)
    fns = [int(i.replace('iteration','')) for i in fns if i.startswith('iteration') and 'txt' not in i and '.pdf' not in i]
    p_dict = nested_dict(2, list)
    for n,f in enumerate(sorted(fns)):
        if n > args.max_iterations: continue
        out_ls = os.listdir('{}/iteration{}'.format(args.predict_dir, f))
        out_ls = [o for o in out_ls if o.endswith('txt2.predict.out')]
        out = '{}/iteration{}/{}'.format(args.predict_dir, f, out_ls[0])
        print('process:',f, out)
        
        out_dict = util.read_icshape_out(out)
        savefn = out+'.vsTrueAtNull'
        p,base_null_count,base_null_predict_count = compare_two_dict(out_dict, validation_dict, savefn)
        p_dict['corr'][f] = p
        p_dict['base_null_count'][f] = base_null_count
        p_dict['base_null_predict_count'][f] = base_null_predict_count
    p_df = pd.DataFrame.from_dict(p_dict)
    print(p_df)
    
    savefn = args.predict_dir + '/corr.txt'
    p_df.to_csv(savefn, header=True, index=True, sep='\t')

def generate_new_shape_out_with_validation_null(args):
    icshape = args.icshape
    out_dict = util.read_icshape_out(icshape)
    
    validation = args.validation
    validation_dict = read_validation_null_true(validation)
    
    new_icshape = icshape+'.newwithNULL'
    for i in validation_dict:
        if i not in out_dict: continue
        for j in validation_dict[i]:
            out_dict[i]['reactivity_ls'][j] = 'NULL'
            
    with open(new_icshape, 'w') as SAVEFN:
        for i,j in out_dict.items():
            SAVEFN.write('\t'.join(map(str, [i, j['length'], '*']+ j['reactivity_ls']))+'\n')
            
def plot_tx_shape_iterations(args):
    icshape_true = args.icshape_true
    out_dict_true = util.read_icshape_out(icshape_true)
    
    icshape = args.icshape
    out_dict = util.read_icshape_out(icshape)
    
    fns = os.listdir(args.predict_dir)
    fns = [int(i.replace('iteration','')) for i in fns if i.startswith('iteration') and 'txt' not in i and '.pdf' not in i]
    
    tx_reactivity_ls = []
    tx_reactivity_ls.append(out_dict_true[args.tx]['reactivity_ls'])
    tx_reactivity_ls.append(out_dict[args.tx]['reactivity_ls'])
    for n,f in enumerate(sorted(fns)):
        if n > args.max_iterations: continue
        out_ls = os.listdir('{}/iteration{}'.format(args.predict_dir, f))
        out_ls = [o for o in out_ls if o.endswith('txt2.predict.out')]
        out = '{}/iteration{}/{}'.format(args.predict_dir, f, out_ls[0])
        print('process:',f, out)
        
        out_dict = util.read_icshape_out(out)
        tx_reactivity_ls.append(out_dict[args.tx]['reactivity_ls'])
        
    tx_reactivity_df = pd.DataFrame(tx_reactivity_ls)
    tx_reactivity_df.replace('NULL', np.nan, inplace=True)
    tx_reactivity_df.replace(-1.0, np.nan, inplace=True)
    tx_reactivity_df.replace('-1', np.nan, inplace=True)
    
    savefn = args.predict_dir + '/sample_tx_shape/{}.txt'.format(args.tx)
    tx_reactivity_df.to_csv(savefn, header=True, index=True, sep='\t')
    print(tx_reactivity_df.head())
    
    fig,ax=plt.subplots(figsize=(30, 10))
    tx_reactivity_df = tx_reactivity_df.applymap(lambda x:float(x))
    sns.heatmap(tx_reactivity_df, cmap='RdYlGn_r', ax=ax)
    plt.tight_layout()
    plt.savefig(savefn.replace('.txt', '.pdf'))
    plt.close()
    
        
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot NULL corr across iterations')
    
    parser.add_argument('--icshape', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80', help='predict file dir')
    parser.add_argument('--predict_dir', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.predict', help='predict file dir')
    parser.add_argument('--validation', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt', help='validation file')
    parser.add_argument('--max_iterations', type=int, default=100, help='plot <= max_iterations')
    parser.add_argument('--tx', type=str, default='ENST00000331434', help='shape plot of tx')
    parser.add_argument('--icshape_true', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80', help='predict file dir')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot NULL corr across iterations', args)
    
    # plot_dir_null_corr(args)
    
    # generate_new_shape_out_with_validation_null(args)
    
    plot_tx_shape_iterations(args)
    
if __name__ == '__main__':
    main()
    
'''
python predict_new_icshape_vs_validate_null_corr.py

python predict_new_icshape_vs_validate_null_corr.py --predict_dir /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.predict --validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt

python predict_new_icshape_vs_validate_null_corr.py --validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt

python predict_new_icshape_vs_validate_null_corr.py --validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt --predict_dir /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.predict 

'''