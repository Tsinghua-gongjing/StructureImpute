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

def read_bed(bed, pureID=1, min_len=10):
    peak_regions = nested_dict(2, list)
    with open(bed, 'r') as BED:
        for line in BED:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            if pureID == 1:
                arr[-3] = arr[-3].split('.')[0]
            peak = '|'.join(arr[-3:])
            score = float(arr[4].split('|')[-1])
            peak_len = int(arr[-1]) - int(arr[-2])
            if peak_len < min_len: continue
            peak_regions[peak]['score'] = score
            peak_regions[peak]['tx'] = arr[-3]
            peak_regions[peak]['start'] = int(arr[-2])
            peak_regions[peak]['end'] = int(arr[-1])
    print('read: {} (peak region n={})'.format(bed, len(peak_regions)))
    return peak_regions.to_dict()

def iclip_has_shape(bed, bed_peak_len, out, max_null_pct=0.4):
    peak_regions = read_bed(bed=bed, pureID=1, min_len=bed_peak_len)
    out_dict = util.read_icshape_out(out=out, pureID=1)
    status_dict = nested_dict(1, int)
    
    BED1 = open(bed.replace('.bed', '.tx_no_shape.bed'), 'w')
    BED2 = open(bed.replace('.bed', '.tx_has_shape_region_exceed_length.bed'), 'w')
    BED3 = open(bed.replace('.bed', '.tx_has_shape_region_null_exceed.bed'), 'w')
    BED4 = open(bed.replace('.bed', '.tx_has_shape_region_null_ok.bed'), 'w')
    
    for i,j in peak_regions.items():
        if j['tx'] not in out_dict:
            status = 'tx_no_shape'
            BED1.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
        else:
            if j['start'] > int(out_dict[j['tx']]['length']) or j['end'] > int(out_dict[j['tx']]['length']):
                status = 'tx_has_shape_region_exceed_length'
                BED2.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
                continue
            reactivity = out_dict[j['tx']]['reactivity_ls'][j['start']:j['end']]
            # print(i, j, reactivity)
            null_pct = reactivity.count('NULL') / len(reactivity)
            if null_pct > max_null_pct:
                status = 'tx_has_shape_region_null_exceed'
                BED3.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
            else:
                status = 'tx_has_shape_region_null_ok'
                BED4.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
        peak_regions[i]['status'] = status
        status_dict[status] += 1
    print('status: ', status_dict)
    BED1.close()
    BED2.close()
    BED3.close()
    BED4.close()
    return status_dict.to_dict()
    
def read_CLIP_table(clip_table_list, cell_line='HEK293|HEK293T'):
    df = pd.read_csv(clip_table_list, header=None, sep='\t')
    columns = ['source_id', 'RBP', 'caller', 'GSE_id', 'cell_line']
    df.columns = columns
    df_select = df[df['cell_line'].isin(cell_line.split('|'))]
    return df_select

def iclip_has_shape_batch(clip_table_list, clip_bed_dir, bed_peak_len, out, max_null_pct):
    df = read_CLIP_table(clip_table_list)
    print(df)
    df_dict = nested_dict(df.to_dict())
    print(df_dict)
    for index,row in df.iterrows():
        bed = '{}/{}_{}_{}_trx.bed'.format(clip_bed_dir, row['source_id'], row['RBP'], row['cell_line'])
        if os.path.isfile(bed):
            pass
        else:
            print('no such file: {}'.format(bed))
            sys.exit()
            
        status_dict = iclip_has_shape(bed=bed, bed_peak_len=bed_peak_len, out=out, max_null_pct=max_null_pct)
        for i,j in status_dict.items():
            df_dict[i][index] = j
    df_dict_new = pd.DataFrame.from_dict(df_dict, orient='columns')
    print(df_dict_new)
    
    savefn = clip_table_list.replace('.tbl', '.select.csv')
    df_dict_new.to_csv(savefn, header=True, index=True, sep='\t')
    
    return savefn
    
def iclip_batch_explore(csv='/home/gongjing/project/shape_imputation/data/CLIP/human.RBP.CLIP.combined.select.csv'):
    df = pd.read_csv(csv, header=0, sep='\t')
    df['#peaks'] = df['tx_has_shape_region_null_exceed'] + df['tx_has_shape_region_null_ok'] + df['tx_no_shape']
    df['#peaks(hasshape)'] = df['tx_has_shape_region_null_exceed'] + df['tx_has_shape_region_null_ok']
    df['ratio'] = df['tx_has_shape_region_null_ok'] / df['#peaks']
    df['ratio(txhasshape)'] = df['tx_has_shape_region_null_ok'] / df['#peaks(hasshape)']
    
    cols = ['#peaks', 'tx_has_shape_region_null_ok', 'tx_has_shape_region_null_exceed', 'tx_no_shape', 'ratio', 'ratio(txhasshape)']
    
    fig,ax=plt.subplots(len(cols),1,figsize=(40,6*len(cols)), sharex=True)

    for n,i in enumerate(cols):
        a = sns.boxplot(x='RBP', y=i, data=df, color=".8", hue='cell_line', ax=ax[n])
        g = sns.stripplot(x='RBP', y=i, data=df, jitter=True, color='black', ax=ax[n])
    ax[n].set_xticklabels(ax[n].xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(csv.replace('.csv', '.pdf'))
    plt.close()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='CLIP data analysis')
    
    parser.add_argument('--clip_bed', type=str, default='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/STARBASE20007_DGCR8_HEK293T_trx.bed', help='Bed file of CLIP data')
    parser.add_argument('--clip_bed_dir', type=str, default='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip', help='Dir to bed file')
    parser.add_argument('--bed_peak_len', type=int, default=10, help='Min peak length to keep')
    parser.add_argument('--icshape', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out', help='icSHAPE out file')
    parser.add_argument('--max_null_pct', type=float, default=0.4, help='Max percentage of NULL values in peak regions')
    parser.add_argument('--clip_table_list', type=str, default='/home/gongjing/project/shape_imputation/data/CLIP/human.RBP.CLIP.combined.tbl', help='All CLIP table list')
    
    # get args
    args = parser.parse_args()
    util.print_args('CLIP data analysis', args)
    
    # read_bed(bed=args.clip_bed)
    # iclip_has_shape(bed=args.clip_bed, bed_peak_len=args.bed_peak_len, out=args.icshape, max_null_pct=args.max_null_pct)
    iclip_has_shape_batch(clip_table_list=args.clip_table_list, clip_bed_dir=args.clip_bed_dir, bed_peak_len=args.bed_peak_len, out=args.icshape, max_null_pct=args.max_null_pct)
    # iclip_batch_explore()
    

if __name__ == '__main__':
    main()
