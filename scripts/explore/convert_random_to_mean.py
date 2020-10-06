import pandas as pd
import numpy as np
import util
import sys,os,random

import argparse
from collections import OrderedDict
import subprocess

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
from nested_dict import nested_dict

def convert(txt, seed=1234):
    random.seed(seed)
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df = pd.read_csv(txt, header=None, sep='\t')
    df.columns = cols
    
    shape_true_dict = nested_dict(1, list)
    for v,s,e in zip(df['fragment_shape(true)'], df['start'], df['end']):
        v_ls = v.split(',')
        for n,i in enumerate(range(s,e)):
            shape_true_dict[i].append(float(v_ls[n]))
    # print(shape_true_dict)
    shape_true = [np.mean(j) for i,j in sorted(shape_true_dict.items(), key=lambda kv: float(kv[0]))]
    shape_true_mean = np.mean([i for i in shape_true if i >= 0])
    shape_true_median = np.percentile([i for i in shape_true if i >= 0], 50)
    
    txt_mean = txt.replace('.txt', '.nullAsMean.txt')
    txt_median = txt.replace('.txt', '.nullAsMedian.txt')
    txt_0 = txt.replace('.txt', '.nullAs0.txt')
    txt_1 = txt.replace('.txt', '.nullAs1.txt')
    txt_random = txt.replace('.txt', '.nullAsRandom.txt')
    df['fragment_shape(nullAsMean)'] = [i.replace('-1', str(shape_true_mean)) for i in df['fragment_shape']]
    df['fragment_shape(nullAsMedian)'] = [i.replace('-1', str(shape_true_median)) for i in df['fragment_shape']]
    df['fragment_shape(nullAs0)'] = [i.replace('-1', str(0)) for i in df['fragment_shape']]
    df['fragment_shape(nullAs1)'] = [i.replace('-1', str(1)) for i in df['fragment_shape']]
    
    random_ls = []
    for i in df['fragment_shape']:
        j = []
        for v in i.split(','):
            if v == '-1':
                j.append(random.uniform(0, 1))
            else:
                j.append(v)
        j_str = ','.join(map(str, j))
        random_ls.append(j_str)
    df['fragment_shape(nullAsRandom)'] = random_ls
    
    convert_ls = ['nullAsMean', 'nullAsMedian', 'nullAs0', 'nullAs1', 'nullAsRandom']
    for c in convert_ls:
        savefn = txt.replace('.txt', '.'+c+'.txt')
        df['fragment_shape({})'.format(c)].to_csv(savefn, header=False, index=False, sep='\t')
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Set random NULL position as other signal')
    
    parser.add_argument('--txt', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL0.3.txt', help='validation data set')
    
    # get args
    args = parser.parse_args()
    util.print_args('Set random NULL position as other signal', args)
    convert(txt=args.txt)
    
if __name__ == '__main__':
    main()