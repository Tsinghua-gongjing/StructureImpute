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

from matplotlib.backends.backend_pdf import PdfPages
import compare_true_and_predict

def read_validate_predict(validate, predict):
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validate = pd.read_csv(validate, header=None, sep='\t')
    df_validate.columns = cols
    
    df_predict = pd.read_csv(predict, header=None, sep='\t')
    df_predict.columns = ['fragment_shape(predict)']
    
    if df_validate.shape[0] != df_predict.shape[0]:
        print('validate{} & predict{} entry num not same'.format(df_validate.shape[0], df_predict.shape[0]))
        sys.exit()
        
    df_validate['fragment_shape(predict)'] = df_predict['fragment_shape(predict)']
    print(df_validate.shape, df_validate.head())
    return df_validate

def compare_predict(validation_ls, predict_ls, label_ls, savefn, bases='AC'):
    validation_ls = validation_ls.split(':')
    predict_ls = predict_ls.split(':')
    label_ls = label_ls.split(':')
    
    df_validate_dict = nested_dict(1, list)
    for validate,predict,label in zip(validation_ls, predict_ls, label_ls):
        df_validate = read_validate_predict(validate, predict)
        df_validate_dict[label] = df_validate
        
    df_validate_merge = df_validate_dict[label_ls[0]].merge(df_validate_dict[label_ls[1]], on=['tx', 'start', 'end'])
    df_validate_merge['corr'] = [stats.pearsonr(list(map(float, i.split(','))),list(map(float, j.split(','))))[0] for i,j in zip(df_validate_merge['fragment_shape(true)_x'], df_validate_merge['fragment_shape(true)_y'])]
    df_validate_merge.sort_values(by=['corr'], inplace=True, ascending=False)
    df_validate_merge['mean_reactivity_x_predict'] = [np.mean([i_v for i_v,j_base in zip(list(map(float, i.split(','))), list(j)) if j_base in bases and i_v>=0]) for i,j in zip(df_validate_merge['fragment_shape(predict)_x'], df_validate_merge['seq_x'])]
    df_validate_merge['mean_reactivity_y_predict'] = [np.mean([i_v for i_v,j_base in zip(list(map(float, i.split(','))), list(j)) if j_base in bases and i_v>=0]) for i,j in zip(df_validate_merge['fragment_shape(predict)_y'], df_validate_merge['seq_y'])]
    print('merge', df_validate_merge.shape, df_validate_merge.head())
    
    pdf = mpl.backends.backend_pdf.PdfPages(savefn)
    
    mean_null_dict = nested_dict(1, list)
    for n,(tx,start,end,s1,s2,s3,s4,s5,s6,seq) in enumerate(zip(df_validate_merge['tx'],df_validate_merge['start'],df_validate_merge['end'],df_validate_merge['fragment_shape_x'],df_validate_merge['fragment_shape(true)_x'],df_validate_merge['fragment_shape(predict)_x'],df_validate_merge['fragment_shape_y'],df_validate_merge['fragment_shape(true)_y'],df_validate_merge['fragment_shape(predict)_y'],df_validate_merge['seq_x'])):
                title = '{}:{}-{}'.format(tx,start,end)
                if n<=100:
                    s3 = ','.join([i if j in 'AC' else '-1' for i,j in zip(s3.split(','),seq)])
                    s6 = ','.join([i if j in 'AC' else '-1' for i,j in zip(s6.split(','),seq)])
                    fig = compare_true_and_predict.plot_bar(shape_ls=[s1,s2,s3,s4,s5,s6], seq=seq, label_ls=['NULL1','True1','Predict1','NULL2','True2','Predict2'], savefn=None, pdf=pdf, title=title, ylim_ls=[[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.1,1.1]])
                mean_null1 = np.mean([j for i,j in zip(list(map(float, s1.split(','))), list(map(float, s2.split(',')))) if i == -1])
                mean_null2 = np.mean([j for i,j in zip(list(map(float, s1.split(','))), list(map(float, s3.split(',')))) if i == -1])
                mean_null3 = np.mean([j for i,j in zip(list(map(float, s4.split(','))), list(map(float, s5.split(',')))) if i == -1])
                mean_null4 = np.mean([j for i,j in zip(list(map(float, s4.split(','))), list(map(float, s6.split(',')))) if i == -1])
                mean_null_dict['mean_null_x'].append(mean_null1)
                mean_null_dict['mean_null_x_predict'].append(mean_null2)
                mean_null_dict['mean_null_y'].append(mean_null3)
                mean_null_dict['mean_null_y_predict'].append(mean_null4)
    for i,j in mean_null_dict.items():
        df_validate_merge[i] = j
                
    plt.close()
    pdf.close()
    
    df_validate_merge.to_csv(savefn.replace('.pdf','.txt'), header=True, index=False, sep='\t')
    
    col_ls = ['mean_reactivity_x', 'mean_reactivity_x_predict', 'mean_reactivity_y', 'mean_reactivity_y_predict']
    df_plot_mean = df_validate_merge.loc[:, col_ls].mean(axis=0)
    
    fig,ax=plt.subplots()
    for i in df_validate_merge.index:
        ax.plot(range(0, len(col_ls)), df_validate_merge.loc[i, col_ls], color='grey', lw=0.8, alpha=0.5)
    ax.plot(range(0, len(col_ls)), df_plot_mean, color='blue', lw=1.2)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf','.mean.pdf'))
    plt.close()
    
    fig,ax=plt.subplots(figsize=(8,8))
    df_validate_merge[col_ls].plot(kind='box')
    r1,p1 = stats.ttest_ind(df_validate_merge['mean_reactivity_x'],df_validate_merge['mean_reactivity_x_predict'])
    r2,p2 = stats.ttest_ind(df_validate_merge['mean_reactivity_y'],df_validate_merge['mean_reactivity_y_predict'])
    r3,p3 = stats.ttest_ind(df_validate_merge['mean_reactivity_x'],df_validate_merge['mean_reactivity_y'])
    r4,p4 = stats.ttest_ind(df_validate_merge['mean_reactivity_x_predict'],df_validate_merge['mean_reactivity_y_predict'])
    title = 'n={}; p1: {:.3f}, p2: {:.3f}, \np3: {:.3f}, p4:{:.3f}'.format(df_validate_merge.shape[0], p1,p2,p3,p4)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf','.mean.box.pdf'))
    plt.close()
    
    col_ls = ['mean_null_x', 'mean_null_x_predict', 'mean_null_y', 'mean_null_y_predict']
    df_plot_mean = df_validate_merge.loc[:, col_ls].mean(axis=0)
    
    fig,ax=plt.subplots()
    for i in df_validate_merge.index:
        ax.plot(range(0, len(col_ls)), df_validate_merge.loc[i, col_ls], color='grey', lw=0.8, alpha=0.5)
    ax.plot(range(0, len(col_ls)), df_plot_mean, color='blue', lw=1.2)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf','.mean.null.pdf'))
    plt.close()
    
    fig,ax=plt.subplots(figsize=(8,8))
    df_validate_merge[col_ls].plot(kind='box')
    r1,p1 = stats.ttest_ind(df_validate_merge['mean_null_x'],df_validate_merge['mean_null_x_predict'])
    r2,p2 = stats.ttest_ind(df_validate_merge['mean_null_y'],df_validate_merge['mean_null_y_predict'])
    r3,p3 = stats.ttest_ind(df_validate_merge['mean_null_x'],df_validate_merge['mean_null_y'])
    r4,p4 = stats.ttest_ind(df_validate_merge['mean_null_x_predict'],df_validate_merge['mean_null_y_predict'])
    # p1 = stats.ks_2samp(df_validate_merge['mean_null_x'],df_validate_merge['mean_null_x_predict'])[1]
    # p2 = stats.ks_2samp(df_validate_merge['mean_null_y'],df_validate_merge['mean_null_y_predict'])[1]
    # p3 = stats.ks_2samp(df_validate_merge['mean_null_x'],df_validate_merge['mean_null_y'])[1]
    # p4 = stats.ks_2samp(df_validate_merge['mean_null_x_predict'],df_validate_merge['mean_null_y_predict'])[1]
    title = 'n={}; p1: {:.3f}, p2: {:.3f}, \np3: {:.3f}, p4:{:.3f}'.format(df_validate_merge.shape[0], p1,p2,p3,p4)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf','.mean.null2.pdf'))
    plt.close()


def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot correlation bar of multiple condition')
    
    parser.add_argument('--validation_ls', type=str, help='Validation file list')
    parser.add_argument('--predict_ls', type=str, help='Predict file list')
    parser.add_argument('--label_ls', type=str, help='Lable list')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/condition_compare_correlation.track.wc_vs_cy.pdf', help='Path to plot file')
    
    # get args
    args = parser.parse_args()
    util.print_args('Plot correlation bar of multiple condition', args)
    compare_predict(validation_ls=args.validation_ls, predict_ls=args.predict_ls, label_ls=args.label_ls, savefn=args.savefn)
    
    # validation_wc = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt'
    # predict_wc = '/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.txt'
    # validation_ch = '/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt'
    # predict_ch = '/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_cy_vivo_0.1.txt'
    # validation_ls = ':'.join([validation_wc, validation_ch])
    # predict_ls = ':'.join([predict_wc, predict_ch])
    # label_ls = 'wc:ch'
    
    # compare_predict(validation_ls=validation_ls, predict_ls=predict_ls, label_ls=label_ls, savefn=args.savefn)

if __name__ == '__main__':
    main()
    
'''
python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt:/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.NULLasWC.txt --predict_ls /home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.txt:/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_cy_vivo_0.1.NULLasWC.txt --label_ls wc:cy_sameNULL --savefn /home/gongjing/project/shape_imputation/results/condition_compare_correlation.track.wc_vs_cy.NULLasWC.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt:/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.NULLasWC.txt --predict_ls /home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.txt:/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_ch_vivo_0.1.NULLasWC.txt --label_ls wc:ch_sameNULL --savefn /home/gongjing/project/shape_imputation/results/condition_compare_correlation.track.wc_vs_ch.NULLasWC.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt:/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.NULLasWC.txt --predict_ls /home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.txt:/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_np_vivo_0.1.NULLasWC.txt --label_ls wc:np_sameNULL --savefn /home/gongjing/project/shape_imputation/results/condition_compare_correlation.track.wc_vs_np.NULLasWC.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt:/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.1.txt:/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_np_vivo0.1.txt --label_ls wc:np --savefn /home/gongjing/project/shape_imputation/results/c80.condition_compare_correlation.track.wc_vs_np.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt:/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.1.txt:/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_ch_vivo0.1.txt --label_ls wc:ch --savefn /home/gongjing/project/shape_imputation/results/c80.condition_compare_correlation.track.wc_vs_ch.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt:/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.1.txt:/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_cy_vivo0.1.txt --label_ls wc:ch --savefn /home/gongjing/project/shape_imputation/results/c80.condition_compare_correlation.track.wc_vs_cy.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt:/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt:/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_np_vivo0.3_trainvalidationinwc6205.txt --label_ls wc:np --savefn /home/gongjing/project/shape_imputation/results/c80.null0.3.condition_compare_correlation.track.wc_vs_np.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt:/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt:/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_cy_vivo0.3_trainvalidationinwc6205.txt --label_ls wc:np --savefn /home/gongjing/project/shape_imputation/results/c80.null0.3.condition_compare_correlation.track.wc_vs_cy.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt:/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt:/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_ch_vivo0.3_trainvalidationinwc6205.txt --label_ls wc:np --savefn /home/gongjing/project/shape_imputation/results/c80.null0.3.condition_compare_correlation.track.wc_vs_ch.pdf

# c94
python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt:/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt:/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_np_vivo0.3_trainvalidationinwc6205.txt --label_ls wc:np --savefn /home/gongjing/project/shape_imputation/results/c94.null0.3.condition_compare_correlation.track.wc_vs_np.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt:/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt:/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_cy_vivo0.3_trainvalidationinwc6205.txt --label_ls wc:np --savefn /home/gongjing/project/shape_imputation/results/c94.null0.3.condition_compare_correlation.track.wc_vs_cy.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt:/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt:/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_ch_vivo0.3_trainvalidationinwc6205.txt --label_ls wc:np --savefn /home/gongjing/project/shape_imputation/results/c94.null0.3.condition_compare_correlation.track.wc_vs_ch.pdf

# d06
python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt:/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vitrorandomNULL0.3.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vivorandomNULL0.3.txt --label_ls DMSseq_K562_vitro:DMSseq_K562_vivo --savefn /home/gongjing/project/shape_imputation/results/d06.randomnull0.3.condition_compare_correlation.track.K562_vitro_vs_vivo.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vitrorandomNULL0.3.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vitrorandomNULL0.3.txt --label_ls DMSseq_K562_vitro:DMSseq_fibroblast_vitro --savefn /home/gongjing/project/shape_imputation/results/d06.randomnull0.3.condition_compare_correlation.track.K562_vitro_vs_fibroblast_vitro.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vitrorandomNULL0.3.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vivorandomNULL0.3.txt --label_ls DMSseq_K562_vitro:DMSseq_fibroblast_vivo --savefn /home/gongjing/project/shape_imputation/results/d06.randomnull0.3.condition_compare_correlation.track.K562_vitro_vs_fibroblast_vivo.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vivorandomNULL0.3.txt:/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vitrorandomNULL0.3.txt --label_ls DMSseq_fibroblast_vivo:DMSseq_fibroblast_vitro --savefn /home/gongjing/project/shape_imputation/results/d06.randomnull0.3.condition_compare_correlation.track.fibroblast_vitro_vs_vivo.pdf

# d10
python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt:/home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vivorandomNULL0.3.txt:/home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vitrorandomNULL0.3.txt --label_ls DMSseq_K562_vivo:DMSseq_K562_vitro --savefn /home/gongjing/project/shape_imputation/results/d10.randomnull0.3.condition_compare_correlation.track.K562_vivo_vs_vitro.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vivorandomNULL0.3.txt:/home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vivorandomNULL0.3.txt --label_ls DMSseq_K562_vivo:DMSseq_fibroblast_vivo --savefn /home/gongjing/project/shape_imputation/results/d10.randomnull0.3.condition_compare_correlation.track.vivo_K562_vs_fibroblast.pdf

python compare_same_seq_true_and_predict_of_two_condition.py --validation_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vitrorandomNULL0.3.txt:/home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vitrorandomNULL0.3.txt --label_ls DMSseq_K562_vivo:DMSseq_K562_vitro --savefn /home/gongjing/project/shape_imputation/results/d10.randomnull0.3.condition_compare_correlation.track.vitro_K562_vs_fibroblast.pdf
'''