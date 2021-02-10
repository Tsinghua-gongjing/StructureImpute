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
import subprocess
import re,random
import torch
import time
from termcolor import colored
import util
import argparse
import null_pattern_shuffle

def data_agumentation(txt=None, times=10, strategy='random'):
    DA_dir = '/'.join(txt.split('/')[0:-1])+'/DA{}'.format(times)
    if not os.path.isdir(DA_dir): os.mkdir(DA_dir)
        
    for i in range(times):
        seed = random.randint(1, 10000000000)
        savefn = DA_dir + '/S{}.txt'.format(seed)
        random.seed(seed)
        if strategy == 'random':
            util.data_random_nullfragament(fragment=txt, null_pct=0.3, col=9, savefn=savefn, mode='randomNperfragment', null_len=5, window_len=100, max_null_len=20)
        if strategy == 'shadow_null_shuffle':
            null_pattern_shuffle.null_pattern_to_bed(txt=txt, seed_ls=[seed])
        
    combine_txt = DA_dir+'.txt'
    cmd = 'cat {}/S* > {}'.format(DA_dir, combine_txt)
    subprocess.call([cmd], shell=True)
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Generate N DA data for a train set')
    
    parser.add_argument('--txt', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment_DA/windowLen100.sliding100.train.txt', help='Path to blastn file')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--times', type=int, default=20, help='DA times')
    parser.add_argument('--strategy', type=str, default='random', help='DA strategy: random|shadow_null_shuffle')
    
    # get args
    args = parser.parse_args()
    util.print_args('Generate N DA data for a train set', args)
    random.seed(args.seed)
    
    data_agumentation(txt=args.txt, times=args.times, strategy=args.strategy)

if __name__ == '__main__':
    main()
    
'''
python generate_train_DA.py --times 2

python generate_train_DA.py --txt /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/DA/DA20/windowLen100.sliding100.train.low_50_1234.null_pattern.txt --strategy shadow_null_shuffle --times 20
'''