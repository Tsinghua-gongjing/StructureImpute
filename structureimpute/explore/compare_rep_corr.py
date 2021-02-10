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
from scipy import stats

def rep_compare(rep1_out=None, rep1_validate=None, rep1_predict=None, rep2_out=None, rep2_validate=None, rep2_predict=None, tx_null_pct=0.3, savefn=None):
    if rep1_out is None: rep1_out = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep1/shape.c200T2M0m0.out'
    if rep1_validate is None: rep1_validate = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep1/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt' 
    if rep1_predict is None: rep1_predict = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep1/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.prediction_trainHasNull_lossAll.txt'
    if rep2_out is None: rep2_out = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep2/shape.c200T2M0m0.out'
    if rep2_validate is None: rep2_validate = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep2/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt' 
    if rep2_predict is None: rep2_predict = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep2/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.prediction_trainHasNull_lossAll.txt'
    if savefn is None: savefn = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/shape_dist/rep.corr.txt'

    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_rep1_validate = pd.read_csv(rep1_validate, header=None, sep='\t')
    df_rep1_validate.columns = cols
    df_rep1_predict = pd.read_csv(rep1_predict, header=None, sep='\t')
    df_rep1_validate['fragment_shape(predict)'] = df_rep1_predict[0]
    
    df_rep2_validate = pd.read_csv(rep2_validate, header=None, sep='\t')
    df_rep2_validate.columns = cols
    df_rep2_predict = pd.read_csv(rep2_predict, header=None, sep='\t')
    df_rep2_validate['fragment_shape(predict)'] = df_rep2_predict[0]
    
    df_rep = df_rep1_validate.merge(df_rep2_validate, how='inner', on=['tx', 'start', 'end'])
    print("df shape: rep1,rep2,rep", df_rep1_validate.shape, df_rep2_validate.shape, df_rep.shape)
    
    out_dict1 = util.read_icshape_out(rep1_out)
    out_dict2 = util.read_icshape_out(rep2_out)
    valid_tx = []
    for i in df_rep['tx']:
        rep1_null_pct = out_dict1[i]['reactivity_ls'].count('NULL') / (float(out_dict1[i]['length'])-35)
        rep2_null_pct = out_dict2[i]['reactivity_ls'].count('NULL') / (float(out_dict2[i]['length'])-35)
        if rep1_null_pct > tx_null_pct: continue
        if rep2_null_pct > tx_null_pct: continue
        valid_tx.append(i)
    print("tx valid number", len(set(valid_tx)))
    
    df_rep = df_rep[df_rep['tx'].isin(valid_tx)]
    df_rep.to_csv(savefn, header=True, index=False, sep='\t')
    
    tx_shape_dict = nested_dict(2, list)
    for tx in set(valid_tx):
        df_tx = df_rep[df_rep['tx']==tx]
        for index,i in df_tx.iterrows():
            for v in i['fragment_shape(true)_x'].split(','): tx_shape_dict[tx]['rep1_before'].append(float(v))
            for v in i['fragment_shape(true)_y'].split(','): tx_shape_dict[tx]['rep2_before'].append(float(v))
            for v in i['fragment_shape(predict)_x'].split(','): tx_shape_dict[tx]['rep1_after'].append(float(v))
            for v in i['fragment_shape(predict)_y'].split(','): tx_shape_dict[tx]['rep2_after'].append(float(v))
                
    corr_dict = nested_dict(2, int)
    for tx in tx_shape_dict:
        v1 = [i for i,j in zip(tx_shape_dict[tx]['rep1_before'], tx_shape_dict[tx]['rep2_before']) if i>=0 and j>=0]
        v2 = [j for i,j in zip(tx_shape_dict[tx]['rep1_before'], tx_shape_dict[tx]['rep2_before']) if i>=0 and j>=0]
        v3 = [i for i,j in zip(tx_shape_dict[tx]['rep1_after'], tx_shape_dict[tx]['rep2_after']) if i>=0 and j>=0]
        v4 = [j for i,j in zip(tx_shape_dict[tx]['rep1_after'], tx_shape_dict[tx]['rep2_after']) if i>=0 and j>=0]
        # print(tx,v1,v2,v3,v4)
        if len(v1) <= 10 or len(v3) <= 10: continue
        c1,p1 = stats.pearsonr(v1, v2)
        c2,p2 = stats.pearsonr(v3, v4)
        corr_dict[tx]['corr_before'] = c1
        corr_dict[tx]['corr_before(p)'] = p1
        corr_dict[tx]['corr_before(n)'] = len(v1)
        corr_dict[tx]['corr_after'] = c2
        corr_dict[tx]['corr_after(p)'] = p2
        corr_dict[tx]['corr_after(n)'] = len(v3)
    corr_df = pd.DataFrame.from_dict(corr_dict, orient='index')
    # print(corr_df)
    corr_df.to_csv(savefn.replace('.pdf', '.txt'), header=True, index=True, sep='\t')
    
    corr_df['# imputated nt'] = corr_df['corr_after(n)'] - corr_df['corr_before(n)']

    fig,ax=plt.subplots(figsize=(8,8))
    sns.scatterplot(x='corr_before', y='corr_after', data=corr_df, ax=ax, hue='# imputated nt')
    ax.set_xlim(0.2,1.05)
    ax.set_ylim(0.2,1.05)
    plt.xticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
        
    
# known_structure_compare()
# rep1_out=None, rep1_validate=None, rep1_predict=None, rep2_out=None, rep2_validate=None, rep2_predict=None, tx_null_pct=0.3, savefn=None

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Compare replicates correlation between true and predicted shape')
    
    parser.add_argument('--rep1_out', type=str, help='Rep1 shape.out')
    parser.add_argument('--rep1_validate', type=str, help='Rep1 fragment')
    parser.add_argument('--rep1_predict', type=str, help='Rep1 fragment prediction')
    parser.add_argument('--rep2_out', type=str, help='Rep2 shape.out')
    parser.add_argument('--rep2_validate', type=str, help='Rep2 fragment')
    parser.add_argument('--rep2_predict', type=str, help='Rep2 fragment prediction')
    parser.add_argument('--tx_null_pct', type=float, default=0.3, help='Cutoff filtering fragment with null pct')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    rep_compare(rep1_out=args.rep1_out, rep1_validate=args.rep1_validate, rep1_predict=args.rep1_predict, 
                rep2_out=args.rep2_out, rep2_validate=args.rep2_validate, rep2_predict=args.rep2_predict, 
                tx_null_pct=args.tx_null_pct, savefn=args.savefn)
    

if __name__ == '__main__':
    main()