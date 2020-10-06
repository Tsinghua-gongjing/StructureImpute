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
import re
import torch
import time
from termcolor import colored
import util
import argparse

def read_blastn(txt):
    fragment_count_dict = nested_dict(1, int)
    fragment_dict = nested_dict(1, list)
    with open(txt, 'r') as TXT:
        for line in TXT:
            line= line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            fragment_count_dict[arr[1]] += 1
            fragment_dict[arr[1]].append(arr[0])
    return fragment_count_dict.to_dict(), fragment_dict.to_dict()

def split(blastn=None, txt=None, validation_pct=0.2):
    fragment_count_dict, fragment_dict = read_blastn(txt=blastn)
    ori = len(fragment_dict)
    fragment_dict_ori = {i:j for i,j in fragment_dict.items()}
    
    validation_num = int(len(fragment_dict) * validation_pct)
    validation_ls_dict = nested_dict(1, int)
    train_ls_dict = nested_dict(1, int)
    for n,(k, v) in enumerate(sorted(fragment_count_dict.items(), key=lambda item: item[1])):
        if k not in fragment_dict: continue
        if n <= validation_num:
            validation_ls_dict[k] += 1
            
            for align in fragment_dict_ori[k]:
                if align != k:
                    # print('del', k, align)
                    if align in fragment_dict: del fragment_dict[align]
        else:
            train_ls_dict[k] += 1
    print('ori: {}, train: {}, validation: {}'.format(ori, len(train_ls_dict), len(validation_ls_dict)))
    
    savefn_train = txt.replace('.txt', '.blastn.train.txt')
    savefn_validation = txt.replace('.txt', '.blastn.validation.txt')
    write_split(txt=txt, train_ls_dict=train_ls_dict, validation_ls_dict=validation_ls_dict, savefn_train=savefn_train, savefn_validation=savefn_validation)

def write_split(txt=None, train_ls_dict=None, validation_ls_dict=None, savefn_train=None, savefn_validation=None):
    SAVEFN_TRAIN = open(savefn_train, 'w')
    SAVEFN_VALIDATE = open(savefn_validation, 'w')
    with open(txt, 'r') as TXT:
        for line in TXT:
            line= line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            fragment = '{}:{}-{}'.format(arr[0], arr[2], arr[3])
            if fragment in train_ls_dict:
                SAVEFN_TRAIN.write(line+'\n')
            if fragment in validation_ls_dict:
                SAVEFN_VALIDATE.write(line+'\n')
                
    SAVEFN_TRAIN.close()
    SAVEFN_VALIDATE.close()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Split into train & validation based on blastn')
    
    parser.add_argument('--blastn', type=str, default='/home/gongjing/project/shape_imputation/data/seq_similarity/windowLen100.sliding10.all2.outputfile_E10', help='Path to blastn file')
    parser.add_argument('--txt', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding10.txt', help='Path to all fragment file')
    parser.add_argument('--validation_pct', type=float, default=0.2, help='Valiation percentage')
    
    # get args
    args = parser.parse_args()
    util.print_args('Split into train & validation based on blastn', args)
    split(blastn=args.blastn, txt=args.txt, validation_pct=args.validation_pct)

if __name__ == '__main__':
    main()