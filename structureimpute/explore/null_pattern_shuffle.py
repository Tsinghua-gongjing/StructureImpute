from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
import sys, os,subprocess
from nested_dict import nested_dict
import pandas as pd
import numpy as np
from pyfasta import Fasta
import os,subprocess
import re
import torch
import time
from termcolor import colored
import util
import argparse
import itertools

def read_null_pattern(null_bed):
    d = nested_dict(2, int)
    with open(null_bed, 'r') as BED:
        for line in BED:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            fragment_start = int(arr[0].split(':')[1].split('-')[0])
            for pos in range(int(arr[1]), int(arr[2])):
                base_pos = pos
                d[arr[0]][base_pos] += 1
    return d.to_dict()

def bed_shuffle(bed, chr_length, seed=1234, chrom=0):
    bedtools = '/home/gongjing/software/bedtools'
    savefn = bed.replace('.bed', '.s{}.chrom{}.bed'.format(seed, chrom))
    if chrom:
        subprocess.call(["{} shuffle -chrom -i {} -g {} -seed {} > {}".format(bedtools, bed, chr_length, seed, savefn)], shell=True)
    else:
        subprocess.call(["{} shuffle -i {} -g {} -seed {} > {}".format(bedtools, bed, chr_length, seed, savefn)], shell=True)
        
    return savefn

def fill_validation_null_with_bed(validation, bed, savefn):
    df = pd.read_csv(validation, header=None, sep='\t')
    columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df.columns = columns
    
    bed_null_d = read_null_pattern(bed)
    
    fragment_shape_ls = []
    for tx,start,end,shape_true in zip(df['tx'], df['start'], df['end'], df['fragment_shape(true)']):
        fragment_id = '{}:{}-{}'.format(tx, start, end)
        v_ls = shape_true.split(',')
        if fragment_id in bed_null_d:
            v_new = []
            for i,v in enumerate(v_ls):
                if i in bed_null_d[fragment_id]:
                    v_new.append('-1')
                else:
                    v_new.append(v)
        else:
            v_new = v_ls
        fragment_shape_ls.append(','.join(v_new))
        
    df['fragment_shape'] = fragment_shape_ls
    df.to_csv(savefn, header=False, index=False, sep='\t')
        
def null_pattern_to_bed(txt, seed_ls=None):
    if seed_ls is None: 
        # seed_ls = [1234]
        seed_ls = ['1234', '40', '9988', '17181790', '81910', '625178', '1', '7829999', '9029102', '918029109']
    bed = txt.replace('.txt', '.bed')
    BED = open(bed, 'w')
    fragment_id_ls = []
    with open(txt, 'r') as TXT:
        for line in TXT:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            fragment_id = '{}:{}-{}'.format(arr[0], arr[2], arr[3])
            start = arr[4]
            end = arr[5]
            BED.write('\t'.join([fragment_id, start, end])+'\n')
            fragment_id_ls.append(fragment_id)
    BED.close()
    
    txt_length = txt.replace('.txt', '.chr.length.txt')
    with open(txt_length, 'w') as TXT:
        for i in list(set(fragment_id_ls)):
            TXT.write('\t'.join([i, '100'])+'\n')
            
    for seed in seed_ls:
        savefn_bed = bed_shuffle(bed=bed, chr_length=txt_length, seed=seed, chrom=0)
        # bed_null_d = read_null_pattern(savefn_bed)
        # print(bed_null_d['ENST00000266970:1500-1600'])

        validation = txt.replace('.null_pattern.txt', '.txt')
        savefn = savefn_bed.replace('.bed', '.txt')
        fill_validation_null_with_bed(validation=validation, bed=savefn_bed, savefn=savefn)
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Generate a shuffle fragment based on a null pattern')
    
    parser.add_argument('--txt', type=str, default='/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_60_1234.null_pattern.txt', help='Path to dir')
    
    args = parser.parse_args()
    util.print_args('Generate a shuffle fragment based on a null pattern', args)
    
    null_pattern_to_bed(txt=args.txt)
    
if __name__ == '__main__':
    main()
    
'''
python null_pattern_shuffle.py --txt /data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_25_1234.null_pattern.txt
python null_pattern_shuffle.py --txt /data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_50_1234.null_pattern.txt
python null_pattern_shuffle.py --txt /data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding10.train.low_50_1234.null_pattern.txt
python null_pattern_shuffle.py --txt /data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding10.blastn.train.low_50_1234.null_pattern.txt
'''