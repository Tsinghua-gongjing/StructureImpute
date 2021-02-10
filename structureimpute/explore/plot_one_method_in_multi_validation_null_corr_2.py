import matplotlib as mpl
#mpl.use('Agg')
mpl.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
import sys, os
from nested_dict import nested_dict
import pandas as pd
import numpy as np
from scipy import stats

import argparse
import util

import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams["font.family"] = "Helvetica"
rc('pdf', use14corefonts=True)

from matplotlib.backends.backend_pdf import PdfPages



def get_df_null(validate, predict, validate_max_null_pct=1, combine_filter_max_null_pct=1, species='human', RNA_type_ls='all'):
    df = pd.read_csv(validate, header=None, sep='\t')
    columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df.columns = columns
    df['fragment_shape'] = [i.replace('NULL','-1') for i in df['fragment_shape']]
    df['null_pct2'] = [(i.split(',').count('-1')+i.split(',').count('NULL'))/float(len(i.split(','))) for i in df['fragment_shape']]
    df = df[df['null_pct2']<=validate_max_null_pct]
    
    df_predict = pd.read_csv(predict, header=None,sep='\t')
    
    if df.shape[0] != df_predict.shape[0]:
        print('validate(n={}) and predict(n={}) not same number'.format(df.shape[0], df_predict.shape[0]))
        sys.exit()
    df['fragment_shape(predict)'] = list(df_predict[0])
    df = df[df['null_pct2']<=combine_filter_max_null_pct]
    df.drop(['null_pct2'], axis=1, inplace=True)
    
    if RNA_type_ls != 'all':
        #print('before filter RNA type: n={}'.format(df.shape[0]))
        trans_dict = util.loadTransGtfBed2(species=species)
        # print(trans_dict)
        RNA_type_ls = RNA_type_ls.split(',')
        df['tx(type)'] = [trans_dict[i]['type'] for i in df['tx']]
        #print(df.head())
        #print(df['tx(type)'].value_counts())
        #print(RNA_type_ls)
        df = df[df['tx(type)'].isin(RNA_type_ls)]
        #print('after filter RNA type: n={}'.format(df.shape[0]))
    
    #print(validate,predict,df.shape,df.head(),df.tail(), df_predict.head(), df_predict.tail())
    
    null_true_ls,null_predict_ls,nonnull_true_ls,nonnull_predict_ls = [],[],[],[]
    null_tx_ls = []
    base_ls = []
    for tx,n,t,p,seq in zip(df['tx'],df['fragment_shape'],df['fragment_shape(true)'],df['fragment_shape(predict)'], df['seq']):
        # print(n,t,p)
        for n_v,t_v,p_v,base in zip(n.split(','),t.split(','),p.split(','), list(seq)):
            if n_v == 'NULL' or float(n_v) == -1:
                null_true_ls.append(float(t_v))
                null_predict_ls.append(float(p_v))
                null_tx_ls.append(tx)
                base_ls.append(base)
            else:
                nonnull_true_ls.append(float(t_v))
                nonnull_predict_ls.append(float(p_v))
    df_null = pd.DataFrame.from_dict({'tx':null_tx_ls,'True':null_true_ls,'Predict':null_predict_ls, 'Base':base_ls})
    
    return df, df_null

def df_tx_corr(df):
    tx_ls = list(set(list(df['tx'])))
    tx_corr_ls = []
    for tx in tx_ls:
        df_tx = df[df['tx']==tx]
        # print(tx, df_tx.shape)
        if df_tx.shape[0] < 10: continue
        tx_r,tx_p = stats.pearsonr(df_tx['True'], df_tx['Predict'])
        if not np.isnan(tx_r):
            tx_corr_ls.append(tx_r)
    return tx_corr_ls

def df_fragment_corr(df):
    df['null_pct2'] = [(i.split(',').count('-1')+i.split(',').count('NULL'))/float(len(i.split(','))) for i in df['fragment_shape']]
    fragment_corr_ls,fragment_corr_p_ls = [],[]
    for i,j in zip(df['fragment_shape(true)'],df['fragment_shape(predict)']):
        r,p = stats.pearsonr(list(map(float,i.split(','))), list(map(float,j.split(','))))
        fragment_corr_ls.append(r)
        fragment_corr_p_ls.append(p)
    df['fragment_corr'] = fragment_corr_ls
    df['fragment_corr_p'] = fragment_corr_p_ls
    df['-log10(p)'] = - np.log10(df['fragment_corr_p'])
    
    return df

def predict_ls_plot(validate_ls, predict_ls, label_ls=None, color_ls=None, savefn=None, validate_max_null_pct=1.0, combine_filter_max_null_pct=1.0, species=None, RNA_type_ls=None, bases='ATCGU'):
    validate_ls = validate_ls.split(',')
    predict_ls = predict_ls.split(',')
    color_ls = color_ls.split(',')
    label_ls = label_ls.split(',')
    
    # trans_dict = util.loadTransGtfBed2(species=species)
    
    df_ls = []
    tx_corr_ls_ls = []
    fig,ax = plt.subplots(1, len(predict_ls), figsize=(5*len(predict_ls), 5))
    for n,predict in enumerate(predict_ls):
        df, df_null = get_df_null(validate=validate_ls[n], predict=predict, validate_max_null_pct=validate_max_null_pct, combine_filter_max_null_pct=combine_filter_max_null_pct, species=species, RNA_type_ls=RNA_type_ls)
        #print('df_null', df_null.head())
        df_null = df_null[df_null['Base'].isin(list(bases))]
        df_null = df_null[df_null['True']>=0]
        df_null.to_csv(savefn.replace('.pdf', '.null.predict_vs_true.txt'), header=True, index=True, sep='\t')
        df = df_fragment_corr(df)
        df_ls.append(df)
        
        tx_corr_ls = df_tx_corr(df_null)
        tx_corr_ls_ls.append(tx_corr_ls)
        
        p = sns.kdeplot(df_null['True'],df_null['Predict'],ax=ax[n], shade=True, color=color_ls[n])
        pearsonr,pval = stats.pearsonr(df_null['True'], df_null['Predict'])
        print("{:.3f} \t{:.3e}".format(pearsonr, pval))
        sys.exit(0)
        ax[n].set_title('pearsonr={:.3f}, p={:.3e}'.format(pearsonr,pval))
        df_null_predict_variance = np.var(df_null['Predict'])
        ax[n].axis('square')
        ax[n].set_xlim(-0.1,1.1)
        ax[n].set_ylim(-0.1,1.1)
        
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    savefn = savefn.replace('.pdf', '.tx_corr.pdf') 
    # util.cumulate_dist_plot(ls_ls=tx_corr_ls_ls,ls_ls_label=label_ls,bins=400,title=None,ax=None,savefn=savefn,xlabel=None,ylabel=None,add_vline=None,add_hline=None,log2transform=0,xlim=None,ylim=None)
    
    df_bed_ls = []
    for i,j in zip(tx_corr_ls_ls, label_ls):
        df_bed = pd.DataFrame.from_dict({'corr':i, 'label':'{}(n={})'.format(j, len(i))})
        df_bed_ls.append(df_bed)
    df_bed_all = pd.concat(df_bed_ls, axis=0)
    # print(df_bed_all)
    
    fig,ax=plt.subplots(figsize=(3*len(label_ls), 6))
    sns.boxplot(x='label',y='corr', data=df_bed_all)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    savefn = savefn.replace('.pdf', '.fragment_corr.pdf') 
    fig,ax = plt.subplots(1, len(predict_ls), figsize=(10*len(predict_ls), 8))
    for n,(df,j) in enumerate(zip(df_ls, label_ls)):
        
        df_plot = df[(df['fragment_corr_p']<=1) &(df['fragment_corr']>0)]
        # cmap = sns.diverging_palette(150, 10, as_cmap=True, center='dark')
        # cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
        cmap  = sns.light_palette("#ED2224", as_cmap=True)
        sns.scatterplot(x='fragment_corr',y='null_pct2',data=df_plot, hue='-log10(p)',ax=ax[n],s=50, palette=cmap)
        ax[n].axvline(x=0.6, ymin=0, ymax=1, color='lightgrey', ls='--')
        ax[n].legend(bbox_to_anchor=(1, 1), loc=2)
        ax[n].set_title('fragement:p<=0.05,r>0 (n={})'.format(df[(df['fragment_corr_p']<=1) &(df['fragment_corr']>0)].shape[0]))
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
        
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot corr of null nucleotides for a list of predicted & validated paired set')
    
    parser.add_argument('--validate_ls', type=str, help='Validate fragment file list')
    parser.add_argument('--predict_ls', type=str, help='Predicted fragment file list')
    parser.add_argument('--color_ls', type=str, default='black,red,green,yellow,grey,cyan', help='Color list')
    parser.add_argument('--savefn', type=str, help='PDF to save plot')
    parser.add_argument('--label_ls', type=str, help='Label for the list')
    parser.add_argument('--validate_max_null_pct', type=float, default=1.0, help='Validation set max NULL')
    parser.add_argument('--combine_filter_max_null_pct', type=float, default=1.0, help='Combine validate and predict, then filter fragment with NULL<=cutoff')
    parser.add_argument('--species', type=str, default='human', help='species')
    parser.add_argument('--RNA_type_ls', type=str, default='all', help='RNA type considered')
    parser.add_argument('--bases', type=str, default='ATCGNU', help='bases considered')
    
    # get args
    args = parser.parse_args()
    
    predict_ls_plot(validate_ls=args.validate_ls, predict_ls=args.predict_ls, color_ls=args.color_ls, savefn=args.savefn, label_ls=args.label_ls, validate_max_null_pct=args.validate_max_null_pct, combine_filter_max_null_pct=args.combine_filter_max_null_pct, species=args.species, RNA_type_ls=args.RNA_type_ls, bases=args.bases)
    

if __name__ == '__main__':
    main()
    
'''
python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_0.1null.txt,/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_wc_vitro_0.1.txt,/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_ch_vivo_0.1.txt,/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_np_vivo_0.1.txt,/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_hek_cy_vivo_0.1.txt,/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_mes_wc_vivo_0.1.txt --savefn /home/gongjing/project/shape_imputation/results/hek293_wc_validation_all_null_corr.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vivo,hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

red: #ED2224
orange: #FF9A01 
green: #119618
pink: #DD4477
blue: #2BADE4
cyan: #9A0099

python plot_one_method_in_multi_validation_null_corr.py --validate /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.txt,/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/hek293_wc_validation_b28_null_corr.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vivo,hek293_wc_vivo


python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA20.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA20.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA20.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA20.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA20.txt --savefn /home/gongjing/project/shape_imputation/results/all_shakerRNA20_null_corr.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA41.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA41.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA41.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA41.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.shaker_predictRNA41.txt --savefn /home/gongjing/project/shape_imputation/results/all_shakerRNA41_null_corr.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_60_1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_60_1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_120Mx1.txt,/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.validation_120Mx1.txt --savefn /home/gongjing/project/shape_imputation/results/b28_on_120M_validateNULL.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vitro,hek293_ch_vivo,hek293_ch_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c48_hek293wc_100M_chrom0shufflenullx10_validationx1/prediction.txt,/home/gongjing/project/shape_imputation/exper/c48_hek293wc_100M_chrom0shufflenullx10_validationx1/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/new_100M_validateNULL.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls test1,test2,test3

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c56_trainpct0.3x10+100Mx10chrom0_validate100M_null0.5/prediction.txt,/home/gongjing/project/shape_imputation/exper/c56_trainpct0.3x10+100Mx10chrom0_validate100M_null0.5/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/new_100M_validateNULL_c56.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls test1,test2,test3 --validate_max_null_pct 0.5 --combine_filter_max_null_pct 1.0

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c52_trainpct0.3x10+100Mx10chrom0_validate100M/prediction.txt,/home/gongjing/project/shape_imputation/exper/c52_trainpct0.3x10+100Mx10chrom0_validate100M/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/new_100M_validateNULL_c52.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls test1,test2,test3 --validate_max_null_pct 1.0 --combine_filter_max_null_pct 0.5

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c55_hek293wc_100M_chrom0shufflenullx10_validationx1_null0.5/prediction.txt,/home/gongjing/project/shape_imputation/exper/c55_hek293wc_100M_chrom0shufflenullx10_validationx1_null0.5/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/new_100M_validateNULL_c55.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls test1,test2,test3 --validate_max_null_pct 0.5 --combine_filter_max_null_pct 1.0

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/new_100M_validateNULL_c80.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls test1,test2,test3 --validate_max_null_pct 1.0 --combine_filter_max_null_pct 1.0


python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.1.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vitro0.1.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_ch_vivo0.1.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_np_vivo0.1.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_cy_vivo0.1.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.mes_wc_vivo0.1.txt --savefn /home/gongjing/project/shape_imputation/results/c80.hek293_wc_validation_all_null_corr.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vivo,hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.1.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vitro0.1_trainvalidation.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_ch_vivo0.1_trainvalidation.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_np_vivo0.1_trainvalidation.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_cy_vivo0.1_trainvalidation.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.mes_wc_vivo0.1_trainvalidation.txt --savefn /home/gongjing/project/shape_imputation/results/c80.hek293_wc_validation_all_null_corr.train+validation.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vivo,hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.inwc6205.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.1.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.1.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vitro0.1_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_ch_vivo0.1_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_np_vivo0.1_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_cy_vivo0.1_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.mes_wc_vivo0.1_trainvalidation.txt --savefn /home/gongjing/project/shape_imputation/results/c80.hek293_wc_validation_all_null_corr.train+validationinwc6205.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vivo,hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_wc_vitro0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_ch_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_np_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.hek_cy_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c80_trainpct0.3x50_validate100M/prediction.mes_wc_vivo0.3_validationin.txt --savefn /home/gongjing/project/shape_imputation/results/c80.hek293_wc_validation_all_null0.3_corr.train+validationinwc6205.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vivo,hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d03_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d03_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/d03.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vivo --bases AC

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low20null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low20null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c99_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull20_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/c99_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull20_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/c99.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vivo --bases AC

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low40null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low40null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d00_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull40_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d00_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull40_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/d00.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vivo --bases AC

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low60null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low60null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d01_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull60_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d01_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull60_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/d01.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vivo --bases AC

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low80null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low80null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d02_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull80_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d02_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull80_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/d02.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vivo --bases AC

# c94
python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.txt,/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/new_100M_validateNULL_c94.pdf --color_ls "#2BADE4,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls test1,test2,test3 --validate_max_null_pct 1.0 --combine_filter_max_null_pct 1.0

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull_randomNULL0.3.inwc6205.txt,/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_wc_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_wc_vitro0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_ch_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_np_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.hek_cy_vivo0.3_trainvalidationinwc6205.txt,/home/gongjing/project/shape_imputation/exper/c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull/prediction.mes_wc_vivo0.3_validationin.txt --savefn /home/gongjing/project/shape_imputation/results/c94.hek293_wc_validation_all_null0.3_corr.train+validationinwc6205.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls hek293_wc_vivo,hek293_wc_vitro,hek293_ch_vivo,hek293_np_vivo,hek293_cy_vivo,mes_wc_vivo

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d04_DMSseq_fibroblast_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d04_DMSseq_fibroblast_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/d04.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vitro,fibroblast_vitro --bases AC

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/d06.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls K562_vitro,K562_vitro --bases AC

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d03_DMSseq_fibroblast_vivo_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d04_DMSseq_fibroblast_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt --savefn /home/gongjing/project/shape_imputation/results/d03-04-06.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vitro,K562_vitro --bases AC

# K562 vitro train and predict others
python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.txt,/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vivo.txt,/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vitro.txt,/home/gongjing/project/shape_imputation/exper/d06_DMSseq_K562_vitro_trainRandmask0.3x50_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vivo.txt --savefn /home/gongjing/project/shape_imputation/results/d06-model.predictOthers.AC.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vitro,K562_vivo,K562_vitro --bases AC

python plot_one_method_in_multi_validation_null_corr.py --validate_ls /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt,/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt,/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.3.txt --predict_ls /home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vitrorandomNULL0.3.txt,/home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_K562_vivorandomNULL0.3.txt,/home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vitrorandomNULL0.3.txt,/home/gongjing/project/shape_imputation/exper/d10_DMSseq_K562_vivo_trainRandmask0.3x10_vallownull100_lossDMSloss_all/prediction.DMSseq_fibroblast_vivorandomNULL0.3.txt --savefn /home/gongjing/project/shape_imputation/results/d10-model.predictOthers.AC.random0.3.pdf --color_ls "#ED2224,#FF9A01,#119618,#DD4477,#2BADE4,#9A0099" --label_ls fibroblast_vivo,fibroblast_vitro,K562_vivo,K562_vitro --bases AC
'''
