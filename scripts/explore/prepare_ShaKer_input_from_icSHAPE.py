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

import util
import argparse

def generate_dbn_react(shape_out=None, species=None, dots=None, savefn_prefix=None, min_len=0):
	dbn = savefn_prefix+'.dbn'
	react = savefn_prefix+'.react'

	shape_dict = util.read_icshape_out(shape_out, pureID=0)
	fa_dict = util.read_fa(fa=None, species=species, pureID=0)
	dot_dict = util.read_dots(dot=dots)


	DBN = open(dbn, 'w')
	REACT = open(react, 'w')
	for i in shape_dict:
		if i in fa_dict:
			seq = fa_dict[i][0:]
		else:
			continue
		if i in dot_dict:
			dot = dot_dict[i]['dotbracket']
		else:
			continue
		print(i, seq, dot, shape_dict[i]['reactivity_ls'])

		if len(seq) < min_len: continue
		if len(set([len(seq),len(dot),len(shape_dict[i]['reactivity_ls'])])) != 1: continue

		DBN.write('>'+i+'\n')
		DBN.write(seq+'\n')
		DBN.write(dot+'\n')
		DBN.write('\n')

		REACT.write('>'+i+'\n')
		for n,v in enumerate(shape_dict[i]['reactivity_ls']):
			REACT.write(str(n+1)+'\t'+v.replace('NULL', 'NA')+'\n')
		REACT.write('\n')

	DBN.close()
	REACT.close()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Generate ShaKer training data from icSHAPE')
    
    parser.add_argument('--shape_out', type=str, default='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_shakerRNA20/3.shape/shape.c200T2M0m0.out', help='icSHAPE out')
    parser.add_argument('--savefn_prefix', type=str, default='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/tools/ShaKer/data/from_icshape/RNA20', help='Save prefix')
    parser.add_argument('--species', type=str, default='RNA20', help='Species')
    parser.add_argument('--dots', type=str, default='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/Shaker_RNA20/RNA20.dot', help='RNA dots file')
    parser.add_argument('--min_len', type=int, default=0, help='Min length considered')


    args = parser.parse_args()
    generate_dbn_react(shape_out=args.shape_out, species=args.species, dots=args.dots, savefn_prefix=args.savefn_prefix, min_len=args.min_len)

if __name__ == '__main__':
    main()

'''
python prepare_ShaKer_input_from_icSHAPE.py

python prepare_ShaKer_input_from_icSHAPE.py --shape_out /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rfam/3.shape/shape.c200T2M0m0.out --savefn_prefix /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/tools/ShaKer/data/from_icshape/Rfam --species human_rfam --dots /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/Rfam_human/human.dot

python prepare_ShaKer_input_from_icSHAPE.py --shape_out /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rfam/3.shape/shape.c200T2M0m0.out --savefn_prefix /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/tools/ShaKer/data/from_icshape/Rfam_minLen90 --species human_rfam --dots /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/Rfam_human/human.dot --min_len 90

python prepare_ShaKer_input_from_icSHAPE.py --shape_out /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out --savefn_prefix /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/tools/ShaKer/data/from_icshape/rRNA18S --species human_rRNA --dots /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/Known_Structures/human_18S.dot2 

'''