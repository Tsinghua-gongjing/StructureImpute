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

def search(species='human', savefn=None):
    fa_dict = util.read_fa(fa=None, species=species, pureID=1)
    with open(savefn, 'w') as SAVEFN:
        p = re.compile(r'[AG][AG]AC[ATC]')
        n = 0
        for i,j in fa_dict.items():
            for m in p.finditer(j[0:]):
                n += 1
#                 SAVEFN.write('\t'.join([i, str(m.span()[0]), str(m.span()[1]), str(n)+'_'+m.group(), '0', '+'])+'\n')
                SAVEFN.write('\t'.join([i, str(m.span()[0]+2), str(m.span()[0]+3), str(n)+'_'+m.group(), '0', '+'])+'\n')
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Search motif from a fasta')
    
    parser.add_argument('--species', type=str, default='human', help='Species')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/hg38_m6A_motif.bed', help='Savefn')
    
    # get args
    args = parser.parse_args()
    util.print_args('Search motif from a fasta', args)
    search(species=args.species, savefn=args.savefn)
    

if __name__ == '__main__':
    main()