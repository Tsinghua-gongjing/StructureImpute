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
from scipy import stats

from matplotlib.backends.backend_pdf import PdfPages

import compare_same_seq_true_and_predict_of_two_condition

def transfer_one_validate_null_to_another(validate1, validate2, savefn):
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validate1 = pd.read_csv(validate1, header=None, sep='\t')
    df_validate1.columns = cols
    
    df_validate2 = pd.read_csv(validate2, header=None, sep='\t')
    df_validate2.columns = cols
    
    df_validate_merge = df_validate1.merge(df_validate2, on=['tx', 'start', 'end'])
    print(df_validate_merge.shape)
    print(df_validate_merge.head())
    
    new_shape_null_ls_ls = []
    for fragment_shape_x,shape_true_y in zip(df_validate_merge['fragment_shape_x'], df_validate_merge['fragment_shape(true)_y']):
        shape_x_ls = fragment_shape_x.split(',')
        shape_true_y_ls = shape_true_y.split(',')
        new_shape_null_ls = []
        for i,j in zip(shape_x_ls, shape_true_y_ls):
            if i == '-1':
                new_shape_null_ls.append('-1')
            else:
                new_shape_null_ls.append(j)
        new_shape_null_ls_ls.append(','.join(new_shape_null_ls))
    df_validate_merge['fragment_shape_y_new'] = new_shape_null_ls_ls
    
    cols_keep = ['tx', 'length_y', 'start', 'end', 'mean_reactivity_y', 'null_pct_y', 'seq_y', 'fragment_shape_y_new', 'fragment_shape(true)_y']
    df_validate_merge[cols_keep].to_csv(savefn, header=False, index=False, sep='\t')
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Set NULL of one validate to another validate file')
    
    parser.add_argument('--validate1', type=str, help='Validation file 1')
    parser.add_argument('--validate2', type=str, help='Validation file 2')
    parser.add_argument('--savefn', type=str, help='Path to tranfered NULL validation file')
    
    # get args
    args = parser.parse_args()
    util.print_args('Set NULL of one validate to another validate file', args)
    
    transfer_one_validate_null_to_another(validate1=args.validate1, validate2=args.validate2, savefn=args.savefn)
    
if __name__ == '__main__':
    main()
    
'''
python set_one_validate_null_to_another_validate.py --validate1 /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt --validate2 /home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --savefn /home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.NULLasWC.txt

python set_one_validate_null_to_another_validate.py --validate1 /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt --validate2 /home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --savefn /home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.NULLasWC.txt

python set_one_validate_null_to_another_validate.py --validate1 /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNperfragmentNullPct0.3.maxL20.S1234.txt --validate2 /home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt --savefn /home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.NULLasWC.txt

'''