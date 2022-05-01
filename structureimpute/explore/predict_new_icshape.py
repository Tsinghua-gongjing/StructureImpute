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
import os, subprocess
import re
import torch
import time
from termcolor import colored
import util
import argparse
import plot_two_shape_common_tx_pct

def complete_shape_out(icshape=None, species_fa=None, species='human', predict_label=None, predict_model=None, pct=0.5, window_len=100, sliding=50, output_dir=None, gpu_id=1):
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    fa_dict = util.read_fa(fa=species_fa, species=species, pureID=1)
    
    icshape_fragment_all = output_dir+'/'+'allfragment.txt'
    icshape_fragment_all2 = icshape_fragment_all+'2'
    util.shape_fragmentation(out=icshape, fa_dict=fa_dict, savefn=icshape_fragment_all, window_len=window_len, sliding=sliding, all_valid_reactivity=0, null_pct_max=2)
    cmd = '''awk '{print $0"\\t"$NF}' ''' + " {} > {}; sed -i 's/NULL/-1/g' {}".format(icshape_fragment_all, icshape_fragment_all2, icshape_fragment_all2)
    # print(cmd)
    subprocess.call([cmd], shell=True)
    
    icshape_fragment_pct = output_dir+'/'+'allfragment.{}.txt'.format(pct)
    util.shape_fragmentation(out=icshape, fa_dict=fa_dict, savefn=icshape_fragment_pct, window_len=window_len, sliding=sliding, all_valid_reactivity=0, null_pct_max=pct)
    icshape_fragment_pct2 = icshape_fragment_pct+'2'
    cmd = '''awk '{print $0"\\t"$NF}' ''' + " {} > {}; sed -i 's/NULL/-1/g' {}".format(icshape_fragment_pct, icshape_fragment_pct2, icshape_fragment_pct2)
    # print(cmd)
    subprocess.call([cmd], shell=True)
    
    predict = output_dir+'/'+'predict.txt'
    cmd_predict = 'bash structureimpute/explore/predict_new_icshape.sh {} {} {} {}'.format(gpu_id, icshape_fragment_pct2, predict, predict_model)
    print(cmd_predict)
    subprocess.call([cmd_predict], shell=True)
    # predict = '/home/gongjing/project/shape_imputation/exper/{}/prediction.{}.txt'.format(predict_model, predict_label)
    
    predict_shape_out = predict.replace('predict.txt', 'predict.out')
    util.predict_to_shape(validation=icshape_fragment_pct2, predict=predict, shape_out=predict_shape_out)
    
    # 从总的fragment减去小于pct的，得到剩余的不用预测的大于pct的fragment
    icshape_fragment_exceed_pct2 = output_dir+'/'+'allfragment.exceed{}.txt2'.format(pct)
    cmd = ''' awk 'NR==FNR{a[$1$3$4];next} !($1$3$4 in a){print $0}' ''' + '''{} {} > {}'''.format(icshape_fragment_pct2, icshape_fragment_all2, icshape_fragment_exceed_pct2)
    subprocess.call([cmd], shell=True)
    
    # 对大于pct的fragment生成预测值文件：直接使用原来的值（即最后一列即可）
    icshape_fragment_exceed_pct2_predict = icshape_fragment_exceed_pct2+'.predict' # not truly predict but generate a pseudo file
    cmd = ''' awk '{print $NF}' ''' + ''' {} > {} '''.format(icshape_fragment_exceed_pct2, icshape_fragment_exceed_pct2_predict)
    subprocess.call([cmd], shell=True)
    
    # 重新合并小于pct和大于pct的fragment文件(相当于validation文件)
    icshape_fragment_pct_plus_exceed_pct2 = output_dir+'/'+'allfragment.{}+exceed{}.txt2'.format(pct, pct)
    cmd = ''' cat {} {} > {}'''.format(icshape_fragment_pct2, icshape_fragment_exceed_pct2, icshape_fragment_pct_plus_exceed_pct2)
    subprocess.call([cmd], shell=True)
    
    # 合并预测的文件
    icshape_fragment_pct_plus_exceed_predict = output_dir+'/'+'allfragment.{}+exceed{}.txt2.predict'.format(pct, pct)
    cmd = ''' cat {} {} > {} '''.format(predict, icshape_fragment_exceed_pct2_predict, icshape_fragment_pct_plus_exceed_predict)
    subprocess.call([cmd], shell=True)
    
    # 根据重新合并的validation，预测文件，生成.out文件
    icshape_fragment_pct_plus_exceed_predict_shapeout = icshape_fragment_pct_plus_exceed_predict+'.out'
    util.predict_to_shape(validation=icshape_fragment_pct_plus_exceed_pct2, predict=icshape_fragment_pct_plus_exceed_predict, shape_out=icshape_fragment_pct_plus_exceed_predict_shapeout)
    
    # 画真实和预测的null pct scatter
    savefn = icshape_fragment_pct_plus_exceed_predict_shapeout+'.scatter.pdf'
    stat1,stat2 = plot_two_shape_common_tx_pct.plot_shape_tx_null_pct(out1=icshape, out2=icshape_fragment_pct_plus_exceed_predict_shapeout, out1_label='True', out2_label='Predict', savefn=savefn, species_fa=species_fa, species=species)
    
    return icshape_fragment_pct_plus_exceed_predict_shapeout,stat1,stat2

def complete_shape_out_nullpct(icshape, species_fa, species, predict_label, predict_model, pct, window_len, sliding, shape_null_pct, gpu_id):
    icshape_predict_dir = icshape+'.predict'
    if not os.path.isdir(icshape_predict_dir): os.mkdir(icshape_predict_dir)
        
    stat_iteration = nested_dict(2, list)
    
    n = 1
    new_shapeout,stat1,stat2 = complete_shape_out(icshape, species_fa, species, predict_label, predict_model, pct, window_len, sliding, output_dir=icshape_predict_dir+'/iteration'+str(n), gpu_id=gpu_id)
    stat_iteration[0] = stat1; stat_iteration[1] = stat2
    while stat2['total_bases(NULL_pct)'] >= shape_null_pct:
        n += 1
        new_shapeout,stat1,stat2 = complete_shape_out(new_shapeout, species_fa, species, predict_label, predict_model, pct, window_len, sliding, output_dir=icshape_predict_dir+'/iteration'+str(n), gpu_id=gpu_id)
        stat_iteration[n] = stat2
    stat_df = pd.DataFrame.from_dict(stat_iteration, orient='columns')
    print(stat_df)
    savefn = icshape_predict_dir+'/iteration.stat.txt'
    stat_df.to_csv(savefn, header=True, index=True, sep='\t')
        
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Complete a input shape.out')
    
    parser.add_argument('--icshape', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out', help='icSHAPE out file')
    parser.add_argument('--species_fa', type=str, default=None, help='Species .fa reference file')
    parser.add_argument('--species', type=str, default='human', help='Species')
    parser.add_argument('--predict_label', type=str, default='wc_all_fragment', help='Predict_label')
    parser.add_argument('--predict_model', type=str, default='/root/StructureImpute/data/meta_model.pt', help='Model used to predict')
    parser.add_argument('--pct', type=float, default=0.5, help='Max NULL percentage in fragment to predict')
    parser.add_argument('--window_len', type=int, default=100, help='window_len')
    parser.add_argument('--sliding', type=int, default=10, help='sliding')
    parser.add_argument('--shape_null_pct', type=float, default=0.3, help='Stop predict when remains pct(NULL) <= cutoff')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU id')
    
    # get args
    args = parser.parse_args()
    util.print_args('Complete a input shape.out', args)
    complete_shape_out_nullpct(icshape=args.icshape, species_fa=args.species_fa, species=args.species, predict_label=args.predict_label, predict_model=args.predict_model, pct=args.pct, window_len=args.window_len, sliding=args.sliding, shape_null_pct=args.shape_null_pct, gpu_id=args.gpu_id)
    
    # new_shapeout = complete_shape_out(icshape=args.icshape, species=args.species, predict_label=args.predict_label, predict_model=args.predict_model, pct=args.pct, window_len=args.window_len, sliding=args.sliding)
    # for i in range(10):
        # new_shapeout = complete_shape_out(icshape=new_shapeout, species=args.species, predict_label=args.predict_label, predict_model=args.predict_model, pct=args.pct, window_len=args.window_len, sliding=args.sliding)
    

if __name__ == '__main__':
    main()
    
'''
python predict_new_icshape.py --icshape /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out
python predict_new_icshape.py --icshape /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80 --predict_model c80_trainpct0.3x50_validate100M

python predict_new_icshape.py --icshape /home/gongjing/project/shape_imputation/data/mammalian/hek_np_vivo/reactivity1.out --predict_model c94_trainpct0.3x50_validate100M_monitorvalloss_train_hasnull_validate_hasnull --shape_null_pct 0.1
'''
