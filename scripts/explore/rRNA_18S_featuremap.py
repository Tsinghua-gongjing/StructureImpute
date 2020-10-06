import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
from nested_dict import nested_dict
import argparse
import util
from scipy import stats

import torch
import torch.nn as nn

import rRNA_18S_track_plot

def plot_multi_track(validation_18S=None, p_ls=None, exper_dir=None, start=0, end=2000, savefn=None):
    if validation_18S is None:
        validation_18S='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL0.1.txt'
    if p_ls is None:
        p_ls = [0.3]

    shape_dict = nested_dict()

    for p in p_ls:
        validation_18S = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL{}.txt'.format(p)
        predict_18S ='{}/prediction.18S{}.txt'.format(exper_dir,p)
        # print("predict_18S", predict_18S)
        shape_random,shape_true,shape_predict,roc_auc_list,validpos_roc_auc_list,nullpos_roc_auc_list = rRNA_18S_track_plot.known_structure_compare(dot=None, validate=validation_18S, predict=predict_18S, tx='18S', start=0, savefn=None)
        
        shape_dict[p]['predict']['random'] = shape_random
        shape_dict[p]['predict']['true'] = shape_true
        shape_dict[p]['predict']['predict'] = shape_predict
        shape_dict[p]['predict']['AUC'] = roc_auc_list[1] # [true,predict]
        shape_dict[p]['predict']['AUC(validpos)'] = validpos_roc_auc_list[1] #
        shape_dict[p]['predict']['AUC(nullpos)'] = nullpos_roc_auc_list[1] #
        
        shape_ls = []
        shape_ls.append(shape_true)
        shape_ls.append(shape_random)
        shape_ls.append(shape_predict)
        
        feature_ls = []
        pos_min_256 = 0
        pos_max_256 = 128
        for i in [0,1,2]:
            predict_out = predict_18S.replace('.txt', 'fmap.out{}.txt'.format(i))
            # shape_random,shape_true,shape_predict,roc_auc_list,validpos_roc_auc_list,nullpos_roc_auc_list = rRNA_18S_track_plot.known_structure_compare(dot=None, validate=validation_18S, predict=predict_out, tx='18S', start=0, savefn=None)
            # shape_dict[p]['out{}'.format(i)]['random'] = shape_random
            # shape_dict[p]['out{}'.format(i)]['true'] = shape_true
            # shape_dict[p]['out{}'.format(i)]['predict'] = shape_predict
            # shape_dict[p]['out{}'.format(i)]['AUC'] = roc_auc_list[1] # [true,predict]
            # shape_dict[p]['out{}'.format(i)]['AUC(validpos)'] = validpos_roc_auc_list[1] #
            # shape_dict[p]['out{}'.format(i)]['AUC(nullpos)'] = nullpos_roc_auc_list[1] #
            # shape_ls.append(shape_predict)
            
            df_predict = pd.read_csv(predict_out, header=None, sep='\t')
            features = []
            for i in df_predict[0]:
                for n,j in enumerate(i.split(',')):
                    if n >= pos_min_256 and n <= pos_max_256:
                        features.append(float(j))
            feature_ls.append(features)
            
        
        # for i in shape_ls: print(len(i))
        
        shape_df = pd.DataFrame(shape_ls)
        print(shape_df.shape, shape_df.head())
        feature_df = pd.DataFrame(feature_ls)
        print(feature_df.shape, feature_df.head())
        
        n_ax = shape_df.shape[0] + feature_df.shape[0]
        fig,ax=plt.subplots(2,1,figsize=(200,n_ax*2), sharex=False, sharey=False)
        g1 = sns.heatmap(shape_df, xticklabels=False, yticklabels=False, cmap='summer', ax=ax[0])
        g1.set_facecolor('grey')
        g2 = sns.heatmap(feature_df, xticklabels=False, yticklabels=False, cmap='summer', ax=ax[1])
        g2.set_facecolor('grey')
        savefn = '{}/prediction.18S{}.featuremap.pdf'.format(exper_dir,p)
        plt.savefig(savefn)
        plt.close()
        
def plot_multi_track_specific_fragment(validation_18S=None, p_ls=None, exper_dir=None, row_idx=1):
    if validation_18S is None:
        validation_18S='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL0.1.txt'
    if p_ls is None:
        p_ls = [0.3]

    for p in p_ls:
        predict_18S ='{}/prediction.18S{}.txt'.format(exper_dir,p)
        df_predict = pd.read_csv(predict_18S, header=None, sep='\t')
        
        validate = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL{}.txt'.format(p)
        df = pd.read_csv(validate, header=None, sep='\t')
        cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
        df.columns = cols
        df['fragment_shape(predict)'] = df_predict[0]
        
        shape_random = map(float, df.iloc[row_idx,]['fragment_shape'].split(','))
        shape_true = map(float, df.iloc[row_idx,]['fragment_shape(true)'].split(','))
        shape_predict = map(float, df.iloc[row_idx,]['fragment_shape(predict)'].split(','))

        shape_random = [np.nan if i == -1 else float(i) for i in shape_random]
        shape_true = [np.nan if i == -1 else float(i) for i in shape_true]
        shape_predict = [np.nan if i == -1 else float(i) for i in shape_predict]

        shape_df = pd.DataFrame([shape_true, shape_random, shape_predict])
        shape_df.columns = list(df.iloc[row_idx,]['seq'].replace('T','U'))
        
        feature_fn = predict_18S.replace('.txt', 'fmap.pt')
        t = torch.load(feature_fn)
        
        df_feature1 = pd.DataFrame(t[0][0][row_idx,:,:].cpu().numpy())
        df_feature2 = pd.DataFrame(t[1][0][row_idx,:,:].cpu().numpy())
        df_feature3 = pd.DataFrame(t[2][0][row_idx,:,:].cpu().numpy())
        
        n_ax=1 + len(t)
        figsize_x = 50
        figsize_y = 20

        fig,ax=plt.subplots(n_ax, 1, figsize=(figsize_x,figsize_y), sharex=False, sharey=False, gridspec_kw = {'height_ratios':[3, 25, 25, 25]})
        g1 = sns.heatmap(shape_df, xticklabels=True, yticklabels=False, cmap='summer', ax=ax[0], lw=0)
        g1.set_facecolor('grey')

        g2 = sns.heatmap(df_feature1.T, xticklabels=False, yticklabels=False, cmap='summer', ax=ax[1], lw=0.0)
        g2 = sns.heatmap(df_feature2.T, xticklabels=False, yticklabels=False, cmap='summer', ax=ax[2], lw=0.0)
        g2 = sns.heatmap(df_feature3.T, xticklabels=False, yticklabels=False, cmap='summer', ax=ax[3], lw=0.0)

        savefn = feature_fn.replace('.pt', '.idx{}.pdf'.format(row_idx))
        plt.savefig(savefn)
        plt.close()
        
                
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot multiple tracks along 18S')
    
    parser.add_argument('--exper_dir', type=str, default='/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M_reevaluate', help='Dir of work experiments')
    parser.add_argument('--p_ls', type=str, default='0.3', help='percentage to plot')
    parser.add_argument('--start', type=int, default=0, help='Plot start index')
    parser.add_argument('--end', type=int, default=2000, help='Plot end index')
    parser.add_argument('--row_idx', type=int, default=1, help='Sample index to plot in predict file')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot')

    
    # get args
    args = parser.parse_args()
    
    # plot_multi_track(exper_dir=args.exper_dir, p_ls=args.p_ls.split(','), start=args.start, end=args.end, savefn=args.savefn)
    plot_multi_track_specific_fragment(exper_dir=args.exper_dir, p_ls=args.p_ls.split(','), row_idx=args.row_idx)
    

if __name__ == '__main__':
    main()