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

def sort(shape1=None, shape2=None, value_col1=7):
    savefn1_sort,_ = util.sort_two_shape(shape1=shape1, value_col1=value_col1, shape2=shape2)

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Sort shape by NULL count')
    
    parser.add_argument('--shape1', type=str, default='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.IGF2BP1_11.txt.shape100.txt', help='Path to shape2')
    parser.add_argument('--shape2', type=str, default='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.IGF2BP1_11.txt.shape', help='Path to shape1')
    parser.add_argument('--value_col1', type=int, default=7, help='Which column index to sort') 
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    sort(shape1=args.shape1, shape2=args.shape2, value_col1=args.value_col1)

if __name__ == '__main__':
    main()
    
'''
python sort_shape_by_null_count.py --shape1 /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.IGF2BP1_11.txt.shape100.txt --shape2 /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.IGF2BP1_11.txt.shape --value_col1 7
'''