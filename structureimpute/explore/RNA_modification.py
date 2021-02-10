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

def read_bed(bed, pureID=1, min_len=0):
    peak_regions = nested_dict(2, list)
    with open(bed, 'r') as BED:
        for line in BED:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            if pureID == 1:
                arr[0] = arr[0].split('.')[0]
            peak = '|'.join(arr[0:3])
            peak_len = int(arr[2]) - int(arr[1])
            if peak_len < min_len: continue
            peak_regions[peak]['score'] = arr[4]
            peak_regions[peak]['tx'] = arr[0]
            peak_regions[peak]['start'] = int(arr[1])
            peak_regions[peak]['end'] = int(arr[2])
    print('read: {} (peak region n={})'.format(bed, len(peak_regions)))
    return peak_regions.to_dict()

def modification_has_shape(bed, out):
    peak_regions = read_bed(bed=bed, pureID=1, min_len=0)
    out_dict = util.read_icshape_out(out=out, pureID=1)
    status_dict = nested_dict(1, int)
    
    BED1 = open(bed.replace('.bed', '.tx_no_shape.bed'), 'w')
    BED2 = open(bed.replace('.bed', '.tx_has_shape_base_exceed_length.bed'), 'w')
    BED3 = open(bed.replace('.bed', '.tx_has_shape_base_valid.bed'), 'w')
    BED4 = open(bed.replace('.bed', '.tx_has_shape_base_null.bed'), 'w')
    
    for i,j in peak_regions.items():
        if j['tx'] not in out_dict:
            status = 'tx_no_shape'
            BED1.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
        else:
            if j['start'] > int(out_dict[j['tx']]['length']) or j['end'] > int(out_dict[j['tx']]['length']):
                status = 'tx_has_shape_base_exceed_length'
                BED2.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
                continue
            reactivity = out_dict[j['tx']]['reactivity_ls'][j['start']:j['end']][0]
            if reactivity == 'NULL':
                status = 'tx_has_shape_base_null'
                BED4.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
            else:
                status = 'tx_has_shape_base_valid'
                BED3.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
        peak_regions[i]['status'] = status
        status_dict[status] += 1
    print('status: ', status_dict)
    BED1.close()
    BED2.close()
    BED3.close()
    BED4.close()
    return status_dict.to_dict()

def modification_extend_1_has_shape(bed, out, label='e1'):
    peak_regions = read_bed(bed=bed, pureID=1, min_len=0)
    out_dict = util.read_icshape_out(out=out, pureID=1)
    status_dict = nested_dict(1, int)
    
    BED1 = open(bed.replace('.bed', '.{}.tx_no_shape.bed'.format(label)), 'w')
    BED2 = open(bed.replace('.bed', '.{}.tx_has_shape_base_exceed_length.bed'.format(label)), 'w')
    BED3 = open(bed.replace('.bed', '.{}.tx_has_shape_base_valid.bed'.format(label)), 'w')
    BED4 = open(bed.replace('.bed', '.{}.tx_has_shape_base_null.bed'.format(label)), 'w')
    
    for i,j in peak_regions.items():
        if j['tx'] not in out_dict:
            status = 'tx_no_shape'
            BED1.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
        else:
            if j['start'] > int(out_dict[j['tx']]['length']) or j['end'] > int(out_dict[j['tx']]['length']):
                status = 'tx_has_shape_base_exceed_length'
                BED2.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
                continue
            reactivity = out_dict[j['tx']]['reactivity_ls'][j['start']:j['end']+1][0:2]
            if len(reactivity) <=1: continue
            # print(reactivity)
            if reactivity[0] == 'NULL' or reactivity[1] == 'NULL':
                status = 'tx_has_shape_base_null'
                BED4.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
            elif reactivity[0] != 'NULL' and reactivity[1] != 'NULL':
                status = 'tx_has_shape_base_valid'
                BED3.write('\t'.join([j['tx'], str(j['start']), str(j['end'])])+'\n')
        peak_regions[i]['status'] = status
        status_dict[status] += 1
    print('status: ', status_dict)
    BED1.close()
    BED2.close()
    BED3.close()
    BED4.close()
    return status_dict.to_dict()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='RNA modification site analysis')
    
    parser.add_argument('--modification_bed', type=str, default='/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_PseudoU_site.tran.bed', help='Modification bed file')
    parser.add_argument('--icshape', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out', help='icSHAPE out file')
    parser.add_argument('--label', type=str, default='e1')
    
    # get args
    args = parser.parse_args()
    util.print_args('RNA modification site analysis', args)
    # modification_has_shape(bed=args.modification_bed, out=args.icshape)
    modification_extend_1_has_shape(bed=args.modification_bed, out=args.icshape, label=args.label) # for dms-seq, check xxACx, C=>e1

if __name__ == '__main__':
    main()
    
'''
python RNA_modification.py --modification_bed /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.bed --icshape /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out

python RNA_modification.py --modification_bed /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.bed --icshape /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out --label e1fibroblast
'''