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
import subprocess

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def generate_dot_fa_file(dot, save_dir):
    dot_dict = util.read_dots(dot)
    for i in dot_dict:
        savefn = save_dir + '/' + i.replace('/', '_') + '.fa'
        with open(savefn, 'w') as SAVEFN:
            SAVEFN.write('>{}'.format(i)+'\n')
            SAVEFN.write(dot_dict[i]['seq'])

def generate_dot_shape_file(dot, shape, save_dir):
    dot_dict = util.read_dots(dot)
    out_dict = util.read_icshape_out(out=shape, pureID=0)
    for i in out_dict:
        savefn = save_dir + '/' + i.replace('/', '_') + '.shape'
        with open(savefn, 'w') as SAVEFN:
            for n,v in enumerate(out_dict[i]['reactivity_ls']):
                if v == 'NULL': v = '-999'
                SAVEFN.write(' '.join([str(n+1), v])+'\n')
            
def folder_predict(fa_dir, save_dir):
    files = os.listdir(fa_dir)
    files = [i for i in files if i.endswith('.fa')]
    for i in files[0:]:
        print('predict: {}'.format(i))
        file_path = save_dir + '/' + i
        file_ct = file_path.replace('.fa', '.ct')
        cmd = "/home/gongjing/software/rnastructure/Fold {} {}".format(file_path, file_ct)
        subprocess.call([cmd], shell=True)

def fold_shape_predict(fa_dir, shape_dir):
    fa_files = os.listdir(fa_dir)
    shape_files = os.listdir(shape_dir)
    shape_files = [i for i in shape_files if i.endswith('.shape')]
    for i in shape_files:
        if not i.replace('.shape', '.fa') in fa_files: continue
        file_path_fa = fa_dir + '/' + i.replace('.shape', '.fa')
        file_path_shape = shape_dir + '/' + i
        file_ct = shape_dir + '/' + i.replace('.shape', '.ct')
        cmd = "/home/gongjing/software/rnastructure/Fold --SHAPE {} {} {}".format(file_path_shape, file_path_fa, file_ct)
        print(cmd)
        subprocess.call([cmd], shell=True)
        
def ct_dir_to_dot(ct_dir):
    files = os.listdir(ct_dir)
    files = [i for i in files if i.endswith('.ct')]
    for i in files:
        file_ct = ct_dir + '/' + i
        file_dot = ct_dir + '/' + i.replace('.ct', '.dot')
        cmd = "/home/gongjing/software/rnastructure/ct2dot {} 1 {}".format(file_ct, file_dot)
        subprocess.call([cmd], shell=True)
        
def compare_ct(dot, ct_dir_ls, ct_dir_label_ls, savefn):
    dots_dict = util.read_dots(dot)
    
    ct_dir_ls = ct_dir_ls.split(':')
    ct_dir_label_ls = ct_dir_label_ls.split(':')
    
    file_ls = []
    for ct_dir in ct_dir_ls:
        files = os.listdir(ct_dir)
        files = [i for i in files if i.endswith('.dot')]
        file_ls.append(files)
    file_ls_common = list(set(file_ls[0]).intersection(*file_ls))
#     print(file_ls)
#     print(file_ls_common)
    
    ct_dir_dict = nested_dict(3, list)
    for ct_dir,ct_dir_label in zip(ct_dir_ls, ct_dir_label_ls):
        for file in file_ls_common:
            dot = ct_dir + '/' + file
            dot_dict = util.read_dot2(dot)
            ct_dir_dict[ct_dir_label+'(dot)'][file] = dot_dict['dotbracket']
            ct_dir_dict[ct_dir_label+'(energy)'][file] = dot_dict['energy']
#     print(ct_dir_dict)
    
    ct_dir_df = pd.DataFrame.from_dict(ct_dir_dict, orient='columns')
    ct_dir_df['true(dot)'] = [dots_dict[i.replace('_','/').replace('.dot','')]['dotbracket'] for i in ct_dir_df.index]
    
    pair_dict = {'.':0, '(':1, ')':1}
#     pair_dict = {'.':1, '(':0, ')':0}
    for i in ct_dir_label_ls:
        accuracy_ls = []
        recall_ls = []
        precision_ls = []
        f1_ls = []
        for dot1, dot2 in zip(ct_dir_df['true(dot)'], ct_dir_df[i+'(dot)']):
            class1 = [pair_dict[d] for d in list(dot1)]
            class2 = [pair_dict[d] for d in list(dot2)]
            accuracy = accuracy_score(class1, class2)
            accuracy_ls.append(accuracy)
            recall = recall_score(class1, class2)
            recall_ls.append(recall)
            precision = precision_score(class1, class2)
            precision_ls.append(precision)
            f1 = f1_score(class1, class2)
            f1_ls.append(f1)
        ct_dir_df[i+'(accuracy)'] = accuracy_ls
        ct_dir_df[i+'(recall)'] = recall_ls
        ct_dir_df[i+'(precision)'] = precision_ls
        ct_dir_df[i+'(f1)'] = f1_ls
    
    print(ct_dir_df)
    ct_dir_df.to_csv(savefn, header=True, index=True, sep='\t')
    
    
    metric_ls = ['energy', 'accuracy', 'recall', 'precision', 'f1']
    montage_command_str = "montage "
    for metric in metric_ls:
        ct_dir_energy_label_ls = [i+'({})'.format(metric) for i in ct_dir_label_ls]
        df_plot = ct_dir_df[ct_dir_energy_label_ls]
        df_plot = df_plot.applymap(lambda x:float(x))
        df_plot.sort_values(by=[ct_dir_energy_label_ls[0]], inplace=True)
        df_plot.index = [i.replace('.dot','') for i in df_plot.index]
        print(df_plot)

        fig,ax=plt.subplots(figsize=(15,12))
        df_plot.plot(ax=ax, marker='.')
        plt.xticks(range(0, df_plot.shape[0]), df_plot.index, rotation=90)
        plt.tight_layout()
        plt.savefig(savefn.replace('.txt', '.{}.pdf'.format(metric)))
        plt.close()
        
        fig,ax=plt.subplots(figsize=(4,10))
        df_plot.plot(kind='box', ax=ax)
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.tight_layout()
        plt.savefig(savefn.replace('.txt', '.{}.box.pdf'.format(metric)))
        plt.close()
        
        montage_command_str += savefn.replace('.txt', '.{}.pdf'.format(metric)) + ' '
    montage_command_str += "-mode concatenate {}".format(savefn.replace('.txt','.pdf'))
    subprocess.call(montage_command_str, shell=True)
        
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='RNAstructure tool analysis')
    
    parser.add_argument('--dot', type=str, default='/home/gongjing/project/shape_imputation/data/Known_Structures/human.dot', help='Path to dot file')
    parser.add_argument('--shape', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rfam/3.shape/shape.c200T2M0m0.out', help='Path to shape file')
    parser.add_argument('--shape_imputate', type=str, default='/home/gongjing/project/shape_imputation/exper/95_trainLossall_GmultiplyX_null0.1x10/prediction.rfam.out', help='Path to imputated shape file')
    parser.add_argument('--fa_dir', type=str, default='/home/gongjing/project/shape_imputation/data/Known_Structures/human.dot.fa', help='Path to dir of fa file')
    parser.add_argument('--fold_dir', type=str, default='/home/gongjing/project/shape_imputation/data/Known_Structures/human.dot.fold', help='Path to dir to save ct file')
    parser.add_argument('--fold_shape_dir', type=str, default='/home/gongjing/project/shape_imputation/data/Known_Structures/human.dot.fold_shape', help='Path to dir to save ct file')
    parser.add_argument('--fold_shape_imputated_dir', type=str, default='/home/gongjing/project/shape_imputation/data/Known_Structures/human.dot.fold_imputateshape95', help='Path to dir to save ct file')
    parser.add_argument('--compare_savefn', type=str, default='/home/gongjing/project/shape_imputation/data/Known_Structures/compare_fold/seq_shape_imputate95.txt', help='Path to dir to save ct file')
    
    # get args
    args = parser.parse_args()
    util.print_args('RNAstructure tool analysis', args)
#     generate_dot_fa_file(dot=args.dot, save_dir=args.fa_dir)
    
#     folder_predict(fa_dir=args.fa_dir, save_dir=args.fold_dir)
#     ct_dir_to_dot(ct_dir=args.fold_dir)
    
#     generate_dot_shape_file(dot=args.dot, shape=args.shape, save_dir=args.fold_shape_dir)
#     fold_shape_predict(fa_dir=args.fa_dir, shape_dir=args.fold_shape_dir)
#     ct_dir_to_dot(ct_dir=args.fold_shape_dir)
    
    generate_dot_shape_file(dot=args.dot, shape=args.shape_imputate, save_dir=args.fold_shape_imputated_dir)
    fold_shape_predict(fa_dir=args.fa_dir, shape_dir=args.fold_shape_imputated_dir)
    ct_dir_to_dot(ct_dir=args.fold_shape_imputated_dir)

    compare_ct(dot=args.dot, ct_dir_ls=args.fold_dir+':'+args.fold_shape_dir+':'+args.fold_shape_imputated_dir, ct_dir_label_ls='seq:seq_shape:seq_shape_imputate', savefn=args.compare_savefn)
    

if __name__ == '__main__':
    main()
