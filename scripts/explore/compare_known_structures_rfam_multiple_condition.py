import pandas as pd
import numpy as np
import util

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
from nested_dict import nested_dict

def read_AUC(txt):
    df = pd.read_csv(txt, header=0, sep='\t', index_col=0)
    print(df.head())
    return df

def compare_AUCs(AUC_txt_ls, AUC_label_ls, savefn):
    AUC_txt_ls = AUC_txt_ls.split(':')
    AUC_label_ls = AUC_label_ls.split(':')
    AUC_dict = nested_dict(2, list)
    for AUC_txt, AUC_label in zip(AUC_txt_ls, AUC_label_ls):
        df = read_AUC(txt=AUC_txt)
        for i,v in zip(df.index, df['Predict']):
            AUC_dict[i][AUC_label] = v
        for i,v in zip(df.index, df['True']):
            AUC_dict[i]['True'] = v
    AUC_df = pd.DataFrame.from_dict(AUC_dict, orient='index')
    AUC_df.dropna(how='any', inplace=True)
    AUC_df = AUC_df[['True']+AUC_label_ls]
    # AUC_df = AUC_df[AUC_df['True']>=0.5]
    print(AUC_df)
    
    col_ls = ['True']+AUC_label_ls
    df_plot_mean = AUC_df.loc[:, col_ls].mean(axis=0)
    
    fig,ax=plt.subplots()
    for i in AUC_df.index:
        ax.plot(range(0, len(col_ls)), AUC_df.loc[i, col_ls], color='grey', lw=0.8, alpha=0.5)
    ax.plot(range(0, len(col_ls)), df_plot_mean, color='blue', lw=1.2)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot AUC for single known structure')
    
    parser.add_argument('--AUC_txt_ls', type=str, default='/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate/prediction.rfam.AUCs.txt:/home/gongjing/project/shape_imputation/exper/b92_trainLossall_shapeOnly_x10/prediction.rfam.AUCs.txt:/home/gongjing/project/shape_imputation/exper/b91_trainLossall_seqOnly_x10/prediction.rfam.AUCs.txt', help='AUC file list')
    parser.add_argument('--AUC_label_ls', type=str, default='seq+shape:shape:seq', help='AUC label list')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/condition_compare_AUC_rfam.pdf', help='Pdf file to save plot')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    compare_AUCs(AUC_txt_ls=args.AUC_txt_ls, AUC_label_ls=args.AUC_label_ls, savefn=args.savefn)
    
if __name__ == '__main__':
    main()