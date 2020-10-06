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

# bed2
# ENST00000527779.1       DCUN1D5=ENSG00000137692.11      nonsense_mediated_decay 842     1       276     277     528     529     842

def extract_start_codon_shape(species, shape, savefn, max_null_pct=0.4, extend=50):
    trans_dict = util.loadTransGtfBed2(species=species)
    out_dict = util.read_icshape_out(out=shape, pureID=1)
    fa_dict = util.read_fa(fa=None, species=species, pureID=1)
    
    # start codon
    savefn1 = savefn.replace('.shape', '.start_codon.ok.shape')
    savefn2 = savefn.replace('.shape', '.start_codon.null.shape')
    SAVEFN1 = open(savefn1, 'w')
    SAVEFN2 = open(savefn2, 'w')
    
    tx_with_shape = []
    for i,j in out_dict.items():
        if i not in trans_dict: continue
        if int(trans_dict[i]['utr_5_end']) < extend: continue
        if int(trans_dict[i]['cds_end']) - int(trans_dict[i]['cds_start']) < extend: continue
        tx_with_shape.append(i)
        
        # 0-based
        start = int(trans_dict[i]['utr_5_end']) - 49 - 1
        end = int(trans_dict[i]['cds_start']) + 49
        shape = out_dict[i]['reactivity_ls'][start:end]
        seq = fa_dict[i][start:end]
        null_pct = shape.count('NULL') / len(shape)
        shape_str = ','.join(shape).replace('NULL', '-1')
        if null_pct == 1: continue
        if null_pct >= max_null_pct:
            state = 'null'
            SAVEFN2.write('\t'.join(map(str, [i, len(fa_dict[i][0:]), start, end, '.', '.', seq, shape_str, shape_str]))+'\n')
#             SAVEFN2.write('\t'.join(map(str, [i, start, end]))+'\n')
        else:
            state = 'ok'
            SAVEFN1.write('\t'.join(map(str, [i, len(fa_dict[i][0:]), start, end, '.', '.', seq, shape_str, shape_str]))+'\n')
#             SAVEFN1.write('\t'.join(map(str, [i, start, end]))+'\n')
        
    SAVEFN1.close()
    SAVEFN2.close()
    print('tx_with_shape: {}'.format(len(tx_with_shape)))
    
    savefn1_sort,_ = util.sort_two_shape(shape1=savefn1, value_col1=7, shape2=savefn1)
    savefn2_sort,_ = util.sort_two_shape(shape1=savefn2, value_col1=7, shape2=savefn2)
    
    df1 = util.plot_heatmap(fn=savefn1_sort, savefn=savefn1_sort+'.heatmap.pdf', value_col=7, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black')
    df2 = util.plot_heatmap(fn=savefn2_sort, savefn=savefn2_sort+'.heatmap.pdf', value_col=7, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black')
    df1_mean = list(df1.mean())
    df2_mean = list(df2.mean())
    
    fig,ax=plt.subplots(figsize=(16,8))
    ax.plot(df2_mean, label="%s(n=%s)"%('null',df2.shape[0]), marker='.')
    ax.plot(df1_mean, label="%s(n=%s)"%('ok',df1.shape[0]), marker='.')
    ax.set_ylim(0,0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefn1_sort+'.meta.pdf')
    plt.close()
    
    # stop codon
    savefn1 = savefn.replace('.shape', '.stop_codon.ok.shape')
    savefn2 = savefn.replace('.shape', '.stop_codon.null.shape')
    SAVEFN1 = open(savefn1, 'w')
    SAVEFN2 = open(savefn2, 'w')
    
    tx_with_shape = []
    for i,j in out_dict.items():
        if i not in trans_dict: continue
        if int(trans_dict[i]['utr_3_end']) - int(trans_dict[i]['utr_3_start']) < extend: continue
        if int(trans_dict[i]['cds_end']) - int(trans_dict[i]['cds_start']) < extend: continue
        tx_with_shape.append(i)
        
        # 0-based
        start = int(trans_dict[i]['cds_end']) - 49 - 1
        end = int(trans_dict[i]['utr_3_start']) + 49
        shape = out_dict[i]['reactivity_ls'][start:end]
        null_pct = shape.count('NULL') / len(shape)
        seq = fa_dict[i][start:end]
        shape_str = ','.join(shape).replace('NULL', '-1')
        if null_pct == 1: continue
        if null_pct >= max_null_pct:
            state = 'null'
            SAVEFN2.write('\t'.join(map(str, [i, len(fa_dict[i][0:]), start, end, '.', '.', seq, shape_str, shape_str]))+'\n')
#             SAVEFN2.write('\t'.join(map(str, [i, start, end]))+'\n')
        else:
            state = 'ok'
            SAVEFN1.write('\t'.join(map(str, [i, len(fa_dict[i][0:]), start, end, '.', '.', seq, shape_str, shape_str]))+'\n')
#             SAVEFN1.write('\t'.join(map(str, [i, start, end]))+'\n')
        
    SAVEFN1.close()
    SAVEFN2.close()
    print('tx_with_shape: {}'.format(len(tx_with_shape)))
    
    savefn1_sort,_ = util.sort_two_shape(shape1=savefn1, value_col1=7, shape2=savefn1)
    savefn2_sort,_ = util.sort_two_shape(shape1=savefn2, value_col1=7, shape2=savefn2)
    
    df1 = util.plot_heatmap(fn=savefn1_sort, savefn=savefn1_sort+'.heatmap.pdf', value_col=7, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black')
    df2 = util.plot_heatmap(fn=savefn2_sort, savefn=savefn2_sort+'.heatmap.pdf', value_col=7, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black')
    df1_mean = list(df1.mean())
    df2_mean = list(df2.mean())
    
    fig,ax=plt.subplots(figsize=(16,8))
    ax.plot(df2_mean, label="%s(n=%s)"%('null',df2.shape[0]), marker='.')
    ax.plot(df1_mean, label="%s(n=%s)"%('ok',df1.shape[0]), marker='.')
    ax.set_ylim(0,0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefn1_sort+'.meta.pdf')
    plt.close()
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Extract start/stop condon shape')
    
    parser.add_argument('--species', type=str, default='human', help='human')
    parser.add_argument('--icshape', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out', help='icSHAPE out file')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot', default='/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.shape')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    extract_start_codon_shape(species=args.species, shape=args.icshape, savefn=args.savefn)
    

if __name__ == '__main__':
    main()