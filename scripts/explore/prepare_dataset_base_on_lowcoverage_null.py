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

def set_lowdepth_null(high_validation=None, low_shape_out=None, savefn=None, bases='ATCGUN'):
    low_shape_dict = util.read_icshape_out(out=low_shape_out, pureID=1)
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df = pd.read_csv(high_validation, header=None, sep='\t')
    df.columns = cols
    
    shape_null_ls = []
    for tx,start,end,shape,seq in zip(df['tx'], df['start'], df['end'], df['fragment_shape(true)'], df['seq']):
        if tx not in low_shape_dict:
            shape_null = '.'
            shape_null_ls.append(shape_null)
            continue
        v_ls = shape.split(',')
        v_new_ls = []
        for n,i in enumerate(range(start, end)):
            v = v_ls[n]
            if seq[n] not in bases:
                v_new_ls.append('NULL')
            else:
                if low_shape_dict[tx]['reactivity_ls'][i] == 'NULL':
                    v_new_ls.append('NULL')
                else:
                    v_new_ls.append(v)
        shape_null = ','.join(map(str, v_new_ls))
        shape_null_ls.append(shape_null)
        
    df['fragment_shape'] = shape_null_ls
    high_stat = df.shape
    df = df[df['fragment_shape']!='.']
    high_after_stat = df.shape
    print('high original', high_stat, 'high after', high_after_stat)
    
    df.to_csv(savefn, header=False, index=False, sep='\t')
    subprocess.call(["python data_shape_distribution.py --data {}".format(savefn)], shell=True)
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Set validation data set based on low depth null')
    
    parser.add_argument('--high_validation', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.S1234.txt', help='Path to high depth validation file')
    parser.add_argument('--low_shape_out', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out', help='Path to low depth shape.out data')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/windowLen100.sliding100.train.low100null.txt', help='Path to save file')
    parser.add_argument('--bases', type=str, default='ATCGUN',help='Only consider these bases, for other bases, direct set as NULL')
    
    # get args
    args = parser.parse_args()
    util.print_args('Set validation data set based on low depth null', args)
    set_lowdepth_null(high_validation=args.high_validation, low_shape_out=args.low_shape_out, savefn=args.savefn, bases=args.bases)

if __name__ == '__main__':
    main()
    
    
'''
python prepare_dataset_base_on_lowcoverage_null.py

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt --low_shape_out /home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/windowLen100.sliding100.validation.low100null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt --low_shape_out /home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/windowLen100.sliding100.validation.low100null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.S1234.txt --low_shape_out /home/gongjing/project/shape_imputation/data/hek_wc_vivo_30/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/windowLen100.sliding100.train.low60null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt --low_shape_out /home/gongjing/project/shape_imputation/data/hek_wc_vivo_30/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/windowLen100.sliding100.validation.low60null.txt

# random 10 times
for i in 30 50 60
for i in 25 50 75 100
do
	for seed in 1234 40 9988 17181790 81910 625178 1 7829999 9029102 918029109
	do
    validation=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.S1234.txt
    low_shape=/home/gongjing/project/shape_imputation/data/sampling/hek_wc_vivo_${i}_${seed}/3.shape/shape.c200T2M0m0.out
    savefn=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_${i}_${seed}.txt
    echo $validation
    echo $low_shape
    echo $savefn
    python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn
	done
done

python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn

for i in 30 50 60
# for new
for i in 25 50 75 100
do
	for seed in 1234 40 9988 17181790 81910 625178 1 7829999 9029102 918029109
	do
    validation=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt
    low_shape=/home/gongjing/project/shape_imputation/data/sampling/hek_wc_vivo_${i}_${seed}/3.shape/shape.c200T2M0m0.out
    savefn=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_${i}_${seed}.txt
    echo $validation
    echo $low_shape
    echo $savefn
    python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn
	done
done
python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn


for i in 25 50 75 100
do
	for seed in 1234 40 9988 17181790 81910 625178 1 7829999 9029102 918029109
	do
    validation=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding10.validation_truenull_randomNULL0.3.txt
    low_shape=/home/gongjing/project/shape_imputation/data/sampling/hek_wc_vivo_${i}_${seed}/3.shape/shape.c200T2M0m0.out
    savefn=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding10.validation.low_${i}_${seed}.txt
    echo $validation
    echo $low_shape
    echo $savefn
    python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn
	done
done
python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn

for i in 25 50 75 100
do
	for seed in 1234 40 9988 17181790 81910 625178 1 7829999 9029102 918029109
	do
    validation=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding10.blastn.train_randomNULL0.3.txt
    low_shape=/home/gongjing/project/shape_imputation/data/sampling/hek_wc_vivo_${i}_${seed}/3.shape/shape.c200T2M0m0.out
    savefn=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding10.blastn.train.low_${i}_${seed}.txt
    echo $validation
    echo $low_shape
    echo $savefn
    python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn
	done
done
python prepare_dataset_base_on_lowcoverage_null.py --high_validation $validation --low_shape_out $low_shape --savefn $savefn


python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_fibroblast_vivo_10_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low20null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_fibroblast_vivo_20_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low40null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_fibroblast_vivo_30_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low60null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_fibroblast_vivo_40_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low80null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_fibroblast_vivo_50_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt



python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_fibroblast_vitro_50_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_K562_vitro_50_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt


python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/DMSseq_K562_vivo_50_1234/3.shape/shape.c200T2M0m0.out --bases AC --savefn /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.low100null.txt


# hek_wc_vitro hek_ch_vivo hek_np_vivo hek_cy_vivo mes_wc_vivo
# low100: 这里25是NAI1，NAI2分别sample 50M，总共就是100M，DMSO是其一半
python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/hek_wc_vitro_25_1234/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.low100null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/hek_ch_vivo_25_1234/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.low100null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/hek_np_vivo_25_1234/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.low100null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/hek_cy_vivo_25_1234/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.low100null.txt

python prepare_dataset_base_on_lowcoverage_null.py --high_validation /home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.txt --low_shape_out /home/gongjing/project/shape_imputation/data/sampling/mes_wc_vivo_25_1234/3.shape/shape.c200T2M0m0.out --savefn /home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.low100null.txt
'''