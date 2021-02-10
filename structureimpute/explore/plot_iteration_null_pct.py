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

def plot_iteration(stat_ls=None, label_ls=None, savefn=None, corr_ls=None):
    stat_ls = stat_ls.split(',')
    label_ls = label_ls.split(',')
    if corr_ls: corr_ls = corr_ls.split(',')
    
    iterations = 0
    fig,ax=plt.subplots(figsize=(12, 6))
    for stat,label in zip(stat_ls, label_ls):
        print('process', stat, label)
        df = pd.read_csv(stat, header=0, index_col=0, sep='\t')
        print(df.loc['total_bases(NULL_pct)',])
        
        df_plot = df.loc['total_bases(NULL_pct)',]
        df_plot.plot(ax=ax, label=label, linestyle='-', marker='o', ms=7, color='red')
        # df_plot.plot(kind='scatter', ax=ax, label=label)
        if len(list(df_plot.index)) > iterations:
            iterations = len(list(df_plot.index))
    ax.set_ylim(0,)
    ax.set_ylabel('NULL percentage', color='red')
    ax.set_xlabel('Iterations of imputation')
    iterations = [i for i in list(range(iterations)) if i%5==0]
    plt.xticks(iterations, iterations, rotation=0)
    
    if corr_ls:
        ax2 = ax.twinx()
        for n,c in enumerate(corr_ls):
            df_corr = pd.read_csv(c, header=0, index_col=0, sep='\t')
            df_corr['corr'].plot(ax=ax2, label=label_ls[n], linestyle='-', marker='o', ms=7, color='green')
        ax2.set_ylabel('NULL correlation', color='green')
        ax2.set_ylim(0, 0.7)
        # ax2.legend_.remove()
    
    if len(stat_ls) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    if corr_ls:
        fig,ax=plt.subplots()
        for n,c in enumerate(corr_ls):
            df_corr = pd.read_csv(c, header=0, index_col=0, sep='\t')
            df_corr['predict_pct'] = df_corr['base_null_predict_count']/df_corr['base_null_count']
            df_corr['NULL_pct'] = 1 - df_corr['predict_pct']
            for i in df_corr.index:
                if df_corr.loc[i,'predict_pct'] == 1:
                    idx = i
                    break
            df_corr = df_corr.iloc[0:idx,:]
            print(df_corr)
            ax2 = ax.twinx()
            df_corr['corr'].plot(ax=ax2, label=label_ls[n], linestyle='-', marker='o', ms=7, color='green')
            df_corr.loc[0] = [1]*df_corr.shape[1]
            df_corr = df_corr.sort_index() 
            print(df_corr)
            df_corr['NULL_pct'].plot(ax=ax, label=label_ls[n], linestyle='-', marker='o', ms=7, color='red')
        ax2.set_ylim(0, 0.7)
        ax2.set_ylabel('NULL correlation', color='green')
        ax.set_ylabel('NULL percentage', color='red')
        plt.xticks(df_corr.index, df_corr.index, rotation=0)
        plt.tight_layout()
        plt.savefig(savefn.replace('.pdf','.onlymask.pdf'))
        plt.close()  

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot NULL percentage along iterations')
    
    parser.add_argument('--stat_ls', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/t.out.predict/iteration.stat.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.predict/iteration.stat.txt', help='List of stat file')
    parser.add_argument('--label_ls', type=str, default='test,hek_wc_vivo', help='Label list')
    parser.add_argument('--corr_ls', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.nominus.predict/corr.txt', help='Correlation text list')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/null_pct_interation.pdf', help='Savefn pdf')
    
    args = parser.parse_args()
    util.print_args('Plot NULL percentage along iterations', args)
    
    plot_iteration(stat_ls=args.stat_ls, label_ls=args.label_ls, savefn=args.savefn, corr_ls=args.corr_ls)
    
if __name__ == '__main__':
    main()
    
'''
python plot_iteration_null_pct.py --stat_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.predict/iteration.stat.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.predict/iteration.stat.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.predict/iteration.stat.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.predict/iteration.stat.txt --label_ls hek_wc_vivo,hek_wc_vitro,hek_np_vivo,mes_wc_vivo --savefn /home/gongjing/project/shape_imputation/results/null_pct_interation.pdf


python plot_iteration_null_pct.py --stat_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.nominus.predict/iteration.stat.txt --label_ls hek_wc.out.c80.newwithNULL --corr_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.nominus.predict/corr.txt --savefn /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.nominus.predict/iteration.stat.pdf
'''