import matplotlib as mpl
mpl.use('Agg')
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
plt.rcParams["font.family"] = "Helvetica"

from matplotlib.backends.backend_pdf import PdfPages


def plot_bar(shape_ls, seq, label_ls=None, savefn=None, pdf=None, title=None, ylim_ls=None):
    if label_ls is None: label_ls = ['Random NULL','True','Predict']
    fig,ax=plt.subplots(len(shape_ls),1,sharex=True,sharey=False,figsize=(20,3*len(shape_ls)))
    for n,i in enumerate(shape_ls):
        shape = list(map(float, i.split(',')))
        if n==0 or n==3: color = ['red' if v==-1 else 'lightgrey' for v in shape]
        ax[n].bar(range(len(shape)), shape, align='center',color=color)
        ax[n].set_ylabel(label_ls[n])
        ax[n].set_ylim(ylim_ls[n][0], ylim_ls[n][1])
#     ax[n].set_ylim(-0.1,1.1)
    ax[n].set_xlim(-1,len(shape))
    ax[n].set_xticks(range(0, len(shape)))
    ax[n].set_xticklabels(list(seq))
    ax[0].set_title(title)
    
    if savefn:
        plt.tight_layout()
        plt.savefig(savefn)
    elif pdf:
        pdf.savefig(fig)
        plt.close()
    else:
        return fig
    
def predict_vs_true(df, savefn):
    fragment_corr_ls,fragment_corr_p_ls = [],[]
    for i,j in zip(df['fragment_shape(true)'],df['fragment_shape(predict)']):
        r,p = stats.pearsonr(list(map(float,i.split(','))), list(map(float,j.split(','))))
        fragment_corr_ls.append(r)
        fragment_corr_p_ls.append(p)
    df['fragment_corr'] = fragment_corr_ls
    df['fragment_corr_p'] = fragment_corr_p_ls
    df['-log10(p)'] = - np.log10(df['fragment_corr_p'])
    
    pdf2 = mpl.backends.backend_pdf.PdfPages(savefn)
    
    fig,ax=plt.subplots(figsize=(8,8))
    sns.scatterplot(x='fragment_corr',y='null_pct2',data=df, hue='-log10(p)',ax=ax,s=50)
    plt.axvline(x=0.6, ymin=0, ymax=1, color='lightgrey', ls='--')
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.title('all fragement (n={})'.format(df.shape[0]))
    plt.tight_layout()
    pdf2.savefig(fig)
    
    fig,ax=plt.subplots(figsize=(8,8))
    sns.scatterplot(x='fragment_corr',y='null_pct2',data=df[(df['fragment_corr_p']<=0.05)], hue='-log10(p)',ax=ax,s=50)
    plt.axvline(x=0.6, ymin=0, ymax=1, color='lightgrey', ls='--')
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.title('fragement: p<=0.05 (n={})'.format(df[(df['fragment_corr_p']<=0.05)].shape[0]))
    plt.tight_layout()
    pdf2.savefig(fig)
    
    fig,ax=plt.subplots(figsize=(8,8))
    sns.scatterplot(x='fragment_corr',y='null_pct2',data=df[(df['fragment_corr_p']<=0.05) &(df['fragment_corr']>0)], hue='-log10(p)',ax=ax,s=50)
    plt.axvline(x=0.6, ymin=0, ymax=1, color='lightgrey', ls='--')
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.title('fragement:p<=0.05,r>0 (n={})'.format(df[(df['fragment_corr_p']<=0.05) &(df['fragment_corr']>0)].shape[0]))
    plt.tight_layout()
    pdf2.savefig(fig)
    
    null_true_ls,null_predict_ls,nonnull_true_ls,nonnull_predict_ls = [],[],[],[]
    for n,t,p in zip(df['fragment_shape'],df['fragment_shape(true)'],df['fragment_shape(predict)']):
        for n_v,t_v,p_v in zip(n.split(','),t.split(','),p.split(',')):
            if n_v == 'NULL' or float(n_v) == -1:
                null_true_ls.append(float(t_v))
                null_predict_ls.append(float(p_v))
            else:
                nonnull_true_ls.append(float(t_v))
                nonnull_predict_ls.append(float(p_v))
    df_null = pd.DataFrame.from_dict({'True':null_true_ls,'Predict':null_predict_ls})
    
    fig,ax=plt.subplots(figsize=(8,8))
    sns.scatterplot(x='True',y='Predict',data=df_null, s=10)
    plt.tight_layout()
    pdf2.savefig(fig)
    
    fig,ax=plt.subplots(figsize=(8,8))
    p = sns.kdeplot(df_null['True'],df_null['Predict'],ax=ax, shade=True)
    pearsonr,pval = stats.pearsonr(df_null['True'], df_null['Predict'])
    ax.set_title('pearsonr={:.3f}, p={:.3e}'.format(pearsonr,pval))
    plt.tight_layout()
    pdf2.savefig(fig)
    df_null_predict_variance = np.var(df_null['Predict'])
    print('df_null_predict_variance: {}(n={})'.format(df_null_predict_variance, df_null.shape[0]))
    
    df_nonnull = pd.DataFrame.from_dict({'True':nonnull_true_ls,'Predict':nonnull_predict_ls})
#     fig,ax=plt.subplots(figsize=(6,6))
#     sns.scatterplot(x='True',y='Predict',data=df_nonnull, s=10)
#     plt.tight_layout()
#     pdf2.savefig(fig)
    
    fig,ax=plt.subplots(figsize=(8,8))
    p = sns.kdeplot(df_nonnull['True'],df_nonnull['Predict'],ax=ax, shade=True)
    pearsonr,pval = stats.pearsonr(df_nonnull['True'], df_nonnull['Predict'])
    ax.set_title('pearsonr={:.3f}, p={:.3e}'.format(pearsonr,pval))
    plt.tight_layout()
    pdf2.savefig(fig)
    
    pdf2.close()
    
def compare_plot(fragment=None, predict_type='trainHasNull_lossAll', predict=None, savefn=None, plot_savefn=None):
    df = pd.read_csv(fragment, header=None, sep='\t')
    columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df.columns = columns
    df['fragment_shape'] = [i.replace('NULL','-1') for i in df['fragment_shape']]
    
    df_predict = pd.read_csv(predict, header=None,sep='\t')
    
    df['fragment_shape(predict)'] = df_predict[0]
    df['null_pct2'] = [(i.split(',').count('-1')+i.split(',').count('NULL'))/float(len(i.split(','))) for i in df['fragment_shape']]
    # print(df.head())
    
    # predict_vs_true(df, plot_savefn.replace('.pdf', '.frag_corr.pdf'))
#     return
    
    pdf = mpl.backends.backend_pdf.PdfPages(plot_savefn.replace('.pdf', '.case.pdf'))
    
    diff1_ls,diff2_ls=[],[]
    for tx,start,end,s1,s2,s3,seq in zip(df['tx'],df['start'],df['end'],df['fragment_shape'],df['fragment_shape(true)'],df['fragment_shape(predict)'],df['seq']):
        diff1 = np.mean([abs(float(v2)-float(v3)) for v1,v2,v3 in zip(list(map(float, s1.split(','))),list(map(float, s2.split(','))),list(map(float, s3.split(',')))) if float(v1)==-1])
        diff2 = np.mean([abs(float(v2)-float(v3)) for v1,v2,v3 in zip(list(map(float, s1.split(','))),list(map(float, s2.split(','))),list(map(float, s3.split(',')))) if float(v1)!=-1])
        diff1_ls.append(diff1)
        diff2_ls.append(diff2)
    df['diffmean(null)'] = diff1_ls
    df['diffmean(non-null)'] = diff2_ls
    df.sort_values(by=['diffmean(null)'], inplace=True)
    
    df_head = df.head()
    df_tail = df.tail()
    df_case_head = df[df['null_pct2']==0.05].head(10)
    df_case_tail = df[df['null_pct2']==0.05].tail(10)
    print("df_case shape", df_case_head.shape)
    for tx,start,end,s1,s2,s3,seq in zip(df_head['tx'],df_head['start'],df_head['end'],df_head['fragment_shape'],df_head['fragment_shape(true)'],df_head['fragment_shape(predict)'],df_head['seq']):
        title = '{}:{}-{}'.format(tx,start,end)
        diff = ','.join(map(str,[float(v1)-float(v2) for v1,v2 in zip(list(map(float, s2.split(','))),list(map(float, s3.split(','))))]))
        fig = plot_bar(shape_ls=[s1,s2,s3,diff], seq=seq, label_ls=['Random NULL','True','Predict','Diff'], savefn=None, pdf=pdf, title=title, ylim_ls=[[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.55,0.55]])
    for tx,start,end,s1,s2,s3,seq in zip(df_tail['tx'],df_tail['start'],df_tail['end'],df_tail['fragment_shape'],df_tail['fragment_shape(true)'],df_tail['fragment_shape(predict)'],df_head['seq']):
        title = '{}:{}-{}'.format(tx,start,end)
        diff = ','.join(map(str,[float(v1)-float(v2) for v1,v2 in zip(list(map(float, s2.split(','))),list(map(float, s3.split(','))))]))
        fig = plot_bar(shape_ls=[s1,s2,s3,diff], seq=seq, label_ls=['Random NULL','True','Predict','Diff'], savefn=None, pdf=pdf, title=title, ylim_ls=[[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.55,0.55]])
    for tx,start,end,s1,s2,s3,seq in zip(df_case_head['tx'],df_case_head['start'],df_case_head['end'],df_case_head['fragment_shape'],df_case_head['fragment_shape(true)'],df_case_head['fragment_shape(predict)'],df_case_head['seq']):
        title = 'null:0.05,{}:{}-{}'.format(tx,start,end)
        diff = ','.join(map(str,[float(v1)-float(v2) for v1,v2 in zip(list(map(float, s2.split(','))),list(map(float, s3.split(','))))]))
        fig = plot_bar(shape_ls=[s1,s2,s3,diff], seq=seq, label_ls=['Random NULL','True','Predict','Diff'], savefn=None, pdf=pdf, title=title, ylim_ls=[[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.55,0.55]])
    for tx,start,end,s1,s2,s3,seq in zip(df_case_tail['tx'],df_case_tail['start'],df_case_tail['end'],df_case_tail['fragment_shape'],df_case_tail['fragment_shape(true)'],df_case_tail['fragment_shape(predict)'],df_case_tail['seq']):
        title = 'null:0.05,{}:{}-{}'.format(tx,start,end)
        diff = ','.join(map(str,[float(v1)-float(v2) for v1,v2 in zip(list(map(float, s2.split(','))),list(map(float, s3.split(','))))]))
        fig = plot_bar(shape_ls=[s1,s2,s3,diff], seq=seq, label_ls=['Random NULL','True','Predict','Diff'], savefn=None, pdf=pdf, title=title, ylim_ls=[[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.55,0.55]])
    
    for i in range(10):
        pct = i/10.0
        df_pct = df[(df['null_pct2']>=pct)&(df['null_pct2']<pct+0.1)]
        print(pct,pct+0.1,df_pct.shape)
        
        if df_pct.shape[0] > 100:
            # df_pct = df_pct.sample(10, random_state=1)
            
            fragment_corr_ls,fragment_corr_p_ls = [],[]
            for i,j in zip(df_pct['fragment_shape(true)'],df_pct['fragment_shape(predict)']):
                r,p = stats.pearsonr(list(map(float,i.split(','))), list(map(float,j.split(','))))
                fragment_corr_ls.append(r)
                fragment_corr_p_ls.append(p)
            df_pct['fragment_corr'] = fragment_corr_ls
            df_pct['fragment_corr_p'] = fragment_corr_p_ls
            
            df_pct.sort_values(by='fragment_corr', inplace=True, ascending=False)
            df_pct = df_pct.iloc[0:80,:]
            
        if df_pct.shape[0] > 0:
            for tx,start,end,s1,s2,s3,seq in zip(df_pct['tx'],df_pct['start'],df_pct['end'],df_pct['fragment_shape'],df_pct['fragment_shape(true)'],df_pct['fragment_shape(predict)'],df_pct['seq']):
                title = '{}:{}-{}'.format(tx,start,end)
                diff = ','.join(map(str,[float(v1)-float(v2) for v1,v2 in zip(list(map(float, s2.split(','))),list(map(float, s3.split(','))))]))
                s3 = [s3_val if s3_base in 'AC' else -1 for s3_val,s3_base in zip(list(map(float, s3.split(','))), seq)] # for predict mask non-AC for DMS-seq
                s3 = ','.join(map(str,s3))
                # print(s3)
                fig = plot_bar(shape_ls=[s1,s2,s3,diff], seq=seq, label_ls=['Random NULL','True','Predict','Diff'], savefn=None, pdf=pdf, title=title, ylim_ls=[[-0.1,1.1],[-0.1,1.1],[-0.1,1.1],[-0.55,0.55]])
    plt.close()
    pdf.close()
    return df

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Compare true and predict on validation file')
    
    parser.add_argument('--validate', type=str, help='Validate fragment file')
    parser.add_argument('--predict', type=str, help='Predicted fragment file')
    parser.add_argument('--plot_savefn', type=str, help='Pdf file to save plot')
    
    # get args
    args = parser.parse_args()
    
    df = compare_plot(fragment=args.validate, predict_type='trainHasNull_lossAll', predict=args.predict, savefn=None, plot_savefn=args.plot_savefn)
    

if __name__ == '__main__':
    main()