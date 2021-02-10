import pandas as pd
import numpy as np
import util
import os

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
from nested_dict import nested_dict

    
def plot_dir_loss(d, not_plot_dir_str, only_plot_dir_str, min_loss_col, savefn):
    fn_ls = os.listdir(d)
    for j in not_plot_dir_str.split(':'):
        fn_ls = [i for i in fn_ls if not j in i]
    if only_plot_dir_str != '.':
        fn_ls = [i for i in fn_ls if i in only_plot_dir_str.split(':')]
    fn_ls = [i for i in fn_ls if i != 'readme.txt']
        
    loss_dict = nested_dict()
    loss_ls = []
    for i in fn_ls:
        log = d + '/' + i + '/log.txt'
        
        train_shell = d + '/' + i + '/train.sh'
        model_parameter_dict = {}
        with open(train_shell, 'r') as TRAIN:
            for line in TRAIN:
                line = line.strip('\n')
                line = line.replace('RNA-structure-profile-imputation', 'ShapeImputation')
                if 'CUDA_VISIBLE_DEVICES' in line:
                    continue
                elif '--' not in line:
#                     SAVEFN.write(line+'\n')
                    pass
                else:
                    arr = line.strip(' ').split(' ')
                    arr[1] = '' if len(arr) == 2 else arr[1]
                    model_parameter_dict[arr[0]] = arr[1]
#                     print(arr)
#         print(model_parameter_dict)
        if '--batch_size' not in model_parameter_dict: model_parameter_dict['--batch_size'] = 100
        if '--test_batch_size' not in model_parameter_dict: model_parameter_dict['--test_batch_size'] = 100
        
        if os.path.isfile(log):
            print('process: {}'.format(i))
            log_plot_savefn = log.replace('.txt', '.plot.pdf')
            loss_df = util.read_log(log, savefn=log_plot_savefn, test_batch_size=int(model_parameter_dict['--test_batch_size']))
            if loss_df.shape[0] < 10: continue
            loss_min = loss_df.loc[loss_df[min_loss_col].idxmin()]
            loss_min.loc['epoch',] = loss_min.name
            loss_min.name = i
    #         print(loss_min)
            loss_ls.append(loss_min)
        
    loss_df_all = pd.concat(loss_ls, axis=1)
    loss_df_all = loss_df_all[sorted(loss_df_all.columns)]
    print(loss_df_all)
    savefn_txt = savefn.replace('.pdf', '.csv')
    loss_df_all_T = loss_df_all.T
    loss_df_all_T['epoch'].dtype == 'int'
    loss_df_all_T.to_csv(savefn_txt, header=True, index=True, sep='\t', float_format='%.5f')
    
    fig,ax=plt.subplots(figsize=(max(8,0.5*len(fn_ls)),28))
    cols = ['validate loss (train_nonull_validate_nonull)', 'validate loss (train_hasnull_validate_hasnull)', 'validate loss (train_hasnull_validate_onlynull)', 'validate loss (train_hasnull_validate_nonull)']
    for col in cols:
        ax.plot(loss_df_all.loc[col,], label=col, marker='.')
    for i in range(0, loss_df_all.shape[1]):
        plt.axvline(x=i, ymin=0, ymax=1, ls='--', lw='0.2', color='grey')
    plt.xticks(range(0, len(loss_df_all.columns)), loss_df_all.columns, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot AUC for single known structure')
    
    parser.add_argument('--d', type=str, help='Directory to plot loss', default='/home/gongjing/project/shape_imputation/exper')
    parser.add_argument('--not_plot_dir_str', type=str, help='Not plot dir whose names contain any of the str', default='AllTrue')
    parser.add_argument('--only_plot_dir_str', type=str, help='Only plot dir', default='.')
    parser.add_argument('--min_loss_col', type=str, help='Loss item for finding min loss index', default='validate loss (train_nonull_validate_nonull)')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot')
    
    # get args
    args = parser.parse_args()
    
    plot_dir_loss(d=args.d, not_plot_dir_str=args.not_plot_dir_str, only_plot_dir_str=args.only_plot_dir_str, min_loss_col=args.min_loss_col, savefn=args.savefn)
    

if __name__ == '__main__':
    main()
    
    
"""
python plot_all_model_loss.py --savefn /home/gongjing/project/shape_imputation/results/model_loss.pdf
python plot_all_model_loss.py --min_loss_col "validate loss (train_hasnull_validate_hasnull)" --savefn /home/gongjing/project/shape_imputation/results/model_loss.pdf

"""
