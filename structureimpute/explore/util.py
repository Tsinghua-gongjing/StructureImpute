from __future__ import print_function
from termcolor import colored

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"

import sys, subprocess, shutil, os, time, re

from nested_dict import nested_dict
import pandas as pd
import numpy as np
from pyfasta import Fasta
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

####################################################################
### dir, time, print, frozen
def check_dir_or_make(d):
    if not os.path.exists(d):
        os.makedirs(d)

def timer(start, end, description=''):
    """https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{} {:0>2}:{:0>2}:{:05.2f}".format(description, int(hours), int(minutes), seconds))

def print_args(description, args):
    print(colored("Description: {}".format(description), 'red'))
    for arg in vars(args):
        print('{}{}: {}'.format(' '*4, arg, getattr(args, arg)))

def gini(list_of_values,mode='gini',null_pct=1):
    if len(list_of_values) == 0: return -1
    if list_of_values.count('NULL')/float(len(list_of_values)) > null_pct: return -1
    list_of_values = [i for i in list_of_values if i != 'NULL']
    if len(list_of_values) == 0: return -1
    if type(list_of_values[0]) is str:
        list_of_values = list(map(float,list_of_values))
    if mode == 'mean_reactivity':
        return np.mean(list_of_values)
    if mode == 'gini':
        if sum(list_of_values) == 0: return 0.67
        sorted_list = sorted(list_of_values)
        height, area = 0, 0
        for value in sorted_list:
            height += value
            area += height - value / 2.
        fair_area = height * len(list_of_values) / 2.
        return (fair_area - area) / fair_area
####################################################################
    
####################################################################
### read files
def read_icshape_out(out=None, pureID=1):
	print("read: %s"%(out))
	out_dict = nested_dict()
	with open(out, 'r') as OUT:
		for line in OUT:
			line = line.strip()
			if not line or line.startswith('#'): continue
			arr = line.split('\t')
			tx_id = arr[0]
			if pureID:
				tx_id = tx_id.split('.')[0]
			length = int(arr[1])
			rpkm = float(arr[2]) if arr[2] != '*' else arr[2]
			reactivity_ls = arr[3:]
			out_dict[tx_id]['tx_id'] = tx_id
			out_dict[tx_id]['length'] = length
			out_dict[tx_id]['rpkm'] = rpkm
			out_dict[tx_id]['reactivity_ls'] = reactivity_ls
	return out_dict.to_dict()

def read_fa(fa=None, species='human', pureID=1):
    if fa is None:
        if species == 'mouse':
            fa = '/home/gongjing/project/shape_imputation/data/ref/mm10/mm10_transcriptome.fa'
        if species == 'human':
            fa = '/home/gongjing/project/shape_imputation/data/ref/hg38/hg38_transcriptome.fa'
        if species == 'human_rRNA':
            fa = '/home/gongjing/project/shape_imputation/data/ref/rRNA_human/ribosomalRNA_4.fa'
        if species == 'human_rfam':
            fa = '/home/gongjing/project/shape_imputation/data/ref/Rfam_human/human.dot.dedup.fa'
        if species == 'mouse(CIRSseq)':
            fa = '/home/gongjing/project/shape_imputation/data/CIRSseq/cirs.fa'
        if species == 'human_yubo':
            fa = '/home/gongjing/project/shape_imputation/data/mammalian/transcriptome.fa'
        if species == 'mouse_yubo':
            fa = '/home/gongjing/project/shape_imputation/data/mammalian/mouse_transcriptome.fa'
        if species == 'zebrafish':
            fa = '/root/StructureImpute/data/danRer10.refSeq.transcriptome.fa'
    fa_dict1 = Fasta(fa, key_fn=lambda key:key.split("\t")[0])
    if pureID:
        fa_dict = {i.split()[0].split('.')[0]:j[0:] for i,j in fa_dict1.items()}
    else:
        fa_dict = {i.split()[0]:j[0:] for i,j in fa_dict1.items()}
    print(list(fa_dict.keys())[0:3])
    return fa_dict

def read_rpkm_txt(txt, min_val=-1):
	val_dict = nested_dict()
	gene_ls = []
	with open(txt, 'r') as TXT:
		for line in TXT:
			line = line.strip()
			if not line or line.startswith('#'): continue
			arr = line.split('\t')
			val_dict[arr[0]] = float(arr[4])
			gene_ls.append(arr[0])
	return val_dict,gene_ls

def read_rpkm_rep(rep1=None,rep2=None):
	if rep1 is None: rep1='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/4.rpkm/D1.rpkm'
	if rep2 is None: rep2='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/4.rpkm/D2.rpkm'
	val_dict1,gene_ls1=read_rpkm_txt(rep1)
	val_dict2,gene_ls2=read_rpkm_txt(rep2)
	gene_ls_common = list(set(gene_ls1) & set(gene_ls2))
	gene_ls_common_dict={}
	for i in gene_ls_common:
		gene_ls_common_dict[i.split('.')[0]] = (val_dict1[i] + val_dict2[i])/2.0
	return gene_ls_common_dict

def loadTransGtfBed2(species='human', pureID=1):
    if species == 'human':
        ref_bed = '/home/gongjing/project/shape_imputation/data/ref/hg38/hg38.transCoor.bed.3'
    if species == 'mouse':
        ref_bed = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/mm10/mm10.transCoor.bed.3'
    H = open(ref_bed)
    line = H.readline()
    trans_dict = nested_dict()
    header_ls = ['tx', 'gene', 'type', 'length', 'utr_5_start', 'utr_5_end', 'cds_start', 'cds_end', 'utr_3_start', 'utr_3_end']
    while line:
        if line.startswith('#'): line = H.readline(); continue
        arr = line.strip().split()
        if pureID:
            arr[0] = arr[0].split('.')[0]
        for i,j in zip(header_ls, arr):
            trans_dict[arr[0].split('.')[0]][i] = j
        line = H.readline()
    H.close()
    print("read: %s, n=%s"%(ref_bed, len(trans_dict)))
    return trans_dict.to_dict()

def read_log(log, savefn=None, test_batch_size=None):
    loss_dict = nested_dict(2, int)
    with open(log, 'r') as LOG:
        for line in LOG:
            line= line.strip('\n').rstrip(' ')
#             print(line)
            if not line or line.startswith('#'): continue
            arr = line.split(' ')
#             print(arr)
            if line.startswith('[Train monitor]'):    
                epoch = int(arr[3])
                loss_dict['train loss'][epoch] = float(arr[-1])
            if 'train_nonull_validate_nonull' in line:
                loss_dict['validate loss (train_nonull_validate_nonull)'][epoch] = float(arr[-1])*test_batch_size
            if 'train_hasnull_validate_nonull' in line:
                loss_dict['validate loss (train_hasnull_validate_nonull)'][epoch] = float(arr[-1])*test_batch_size
            if 'train_hasnull_validate_hasnull' in line:
                loss_dict['validate loss (train_hasnull_validate_hasnull)'][epoch] = float(arr[-1])*test_batch_size
            if 'train_hasnull_validate_onlynull' in line:
                loss_dict['validate loss (train_hasnull_validate_onlynull)'][epoch] = float(arr[-1])*test_batch_size
    loss_df = pd.DataFrame.from_dict(loss_dict, orient='columns')
    print(loss_df.head())
    
    if savefn:
        fig,ax=plt.subplots(figsize=(12,6))
        for i in loss_df.columns:
            ax.plot(loss_df[i], label=i)
        plt.legend(bbox_to_anchor=(1, 1), loc=2)
        plt.savefig(savefn, bbox_inches='tight')
    
    return loss_df

def read_dot(dot=None):
    if dot is None: dot = '/home/gongjing/project/shape_imputation/data/Known_Structures/human_18S.dot' 
    dot_dict = nested_dict()
    with open(dot, 'r') as DOT:
        for n,line in enumerate(DOT):
            line = line.strip()
            if not line or line.startswith('#'): continue
            if n == 0:
                dot_dict['name'] = line.replace('>','')
            if n == 1:
                dot_dict['seq'] = line
            if n == 2:
                dot_dict['dotbracket'] = line
    return dot_dict.to_dict()

def read_dot2(dot=None):
    dot_dict = nested_dict()
    with open(dot, 'r') as DOT:
        for n,line in enumerate(DOT):
            line = line.strip()
            if not line or line.startswith('#'): continue
            if n == 0:
                dot_dict['name'] = line.replace('>','')
            if n == 1:
                dot_dict['seq'] = line
            if n == 2:
                dot_dict['dotbracket'] = line
            dot_dict['seq_id'] = dot_dict['name'].split(' ')[-1]
            dot_dict['energy'] = dot_dict['name'].split(' ')[-3]
    return dot_dict.to_dict()

def read_dots(dot=None):
	dot_dict = nested_dict()
	with open(dot, 'r') as DOT:
		for n,line in enumerate(DOT):
			line = line.strip()
			if not line or line.startswith('#'): continue
			if n%3 == 0:
				seq_id = line.replace('>','').split('\t')[0]
				dot_dict[seq_id]['name'] = seq_id
			if n%3 == 1:
				dot_dict[seq_id]['seq'] = line
			if n%3 == 2:
				dot_dict[seq_id]['dotbracket'] = line
	return dot_dict.to_dict()

def read_18S_accessibility(f=None):
    if f is None:
        f = '/home/gongjing/project/shape_imputation/data/Known_Structures/18s_o2_sasa.sort.txt'
    accessibility = pd.read_csv(f, header=None, sep='\t')
    accessibility.columns = ['position', 'base', 'accessibility']
    a_ls = list(accessibility['accessibility'])
    
    return a_ls

def read_RBP_name():
    txt = '/home/gongjing/project/shape_imputation/data/CLIP/motif/RBP.txt'
    df = pd.read_csv(txt, header=None, sep='\t')
    return {i:j for i,j in zip(df[1], df[2])}
####################################################################

####################################################################
### prepare data sets
def shape_fragmentation(out=None, fa_dict=None, savefn=None, window_len=30, sliding=10, all_valid_reactivity=1, null_pct_max=0.9):
    out_dict = read_icshape_out(out)
    with open(savefn, 'w') as SAVEFN:
        for i,j in out_dict.items():
            s = 0
            e = 0
            if int(j['length']) < window_len:
                continue
            else:
                while e < int(j['length']):
                    e = s+window_len
                    if e > int(j['length']):
                        fragment_shape = j['reactivity_ls'][-window_len:]
                        fragment_seq = fa_dict[i][-window_len:]
                        ss,ee = int(j['length'])-window_len,int(j['length'])
                    else:
                        fragment_shape = j['reactivity_ls'][s:e]
                        fragment_seq = fa_dict[i][s:e]
                        ss,ee=s,e
                    mean_reactivity = gini(fragment_shape,mode='mean_reactivity',null_pct=1)
                    null_pct = (fragment_shape.count('NULL')+fragment_shape.count('-1.0')++fragment_shape.count('-1')) / float(window_len)
                    if len(fragment_shape) != window_len: continue
                    if all_valid_reactivity:
                        if any(v in ['NULL', '-1.0', '-1'] for v in fragment_shape):
                            pass
                        else:
							# print>>SAVEFN,'\t'.join(map(str, [i, j['length'], ss, ee, mean_reactivity, null_pct, fragment_seq, ','.join(fragment_shape)]))
                            SAVEFN.write('\t'.join(map(str, [i, j['length'], ss, ee, mean_reactivity, null_pct, fragment_seq, ','.join(fragment_shape)]))+'\n')
                    else:
                        if null_pct <= null_pct_max:
							# print>>SAVEFN,'\t'.join(map(str, [i, j['length'], ss, ee, mean_reactivity, null_pct, fragment_seq, ','.join(fragment_shape)]))
                            SAVEFN.write('\t'.join(map(str, [i, j['length'], ss, ee, mean_reactivity, null_pct, fragment_seq, ','.join(fragment_shape)]))+'\n')
                    s += sliding
                    
def fragment_to_format_data(fragment=None, fragment_len=50, split=0, dataset='validation', feature_size=4, max_null_pct=1):
    if fragment is None:
        fragment = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen50.sliding30.txt'
    df = pd.read_csv(fragment, header=None, sep='\t')
    print(df.head())
    columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
    if dataset == 'validation': columns = columns + ['fragment_shape(true)']
    print(columns)
    df.columns = columns
    print('input origin:', df.shape)
    df['null_pct2'] = [(i.split(',').count('-1')+i.split(',').count('NULL'))/float(len(i.split(','))) for i in df['fragment_shape']]
    df = df[df['null_pct2']<=max_null_pct]
    df.drop(['null_pct2'], axis=1, inplace=True)
    print('input keep pct(null)<={}:'.format(max_null_pct), df.shape)
    
    if split:
        train = fragment.replace('.txt','.train.txt')
        validation = fragment.replace('.txt','.validation.txt')
        df_nonull = df[df['null_pct']==0]
        df_nonull_train = df_nonull.sample(frac=0.7, random_state=1)
        df_nonull_train.to_csv(train, header=None, sep='\t', index=False)
        df_nonull_validation = pd.merge(df_nonull,df_nonull_train,how='outer',indicator=True,on=['tx','start','end']).query('_merge=="left_only"').drop(['_merge'],axis=1)
        df_nonull_validation = df_nonull_validation.iloc[:,0:8]
        df_nonull_validation.columns = columns
        df_nonull_validation['fragment_shape(true)'] = df_nonull_validation['fragment_shape']
        validation_shape = np.array([i.split(',') for i in df_nonull_validation['fragment_shape']])
        random_null = random_mask(x=validation_shape.shape[0],
                                           y=validation_shape.shape[1],
                                           mask_num=int(np.product(validation_shape.shape)*0.1))
        validation_shape[random_null==0] = '-1'
        validation_shape = [','.join(i) for i in validation_shape]
        df_nonull_validation['fragment_shape'] = validation_shape
        df_nonull_validation[columns+['fragment_shape(true)']].to_csv(validation, header=None, sep='\t', index=False)
        
        test = fragment.replace('.txt','.test.txt')
        df[df['null_pct']>0].to_csv(test, header=None, sep='\t', index=False)
        return

    seq_feature = np.ones((df.shape[0], fragment_len, feature_size))
    print(seq_feature.shape,"start encoding RNA sequence feature")
    for n,seq in enumerate(df['seq']):
        # print n,seq
        # seq_feature[n,] = one_hot(seq).flatten() # not work in many testing cases
        seq_feature[n] = encode_rna(seq, feature_size=feature_size)#.flatten()

    shape_value_feature = np.ones((df.shape[0], fragment_len))
    print(shape_value_feature.shape,"start encoding RNA shape value feature")
    for n,fragment_shape in enumerate(df['fragment_shape']):
        # print fragment_shape
        shape_value_feature[n,] = [-1 if i=='NULL' else float(i) for i in fragment_shape.split(',')]

    if dataset == 'validation': 
        shape_true_value_feature = np.ones((df.shape[0], fragment_len))
        for n,fragment_shape in enumerate(df['fragment_shape(true)']):
            shape_true_value_feature[n,] = [-1 if i=='NULL' else float(i) for i in fragment_shape.split(',')]
    
        return seq_feature,shape_value_feature,shape_true_value_feature
    else:
        return seq_feature,shape_value_feature,shape_value_feature

def fragment_split(fragment=None, train_frac=0.7, cols=8):
    if fragment is None:
        fragment = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.txt'
    df = pd.read_csv(fragment, header=None, sep='\t')
    columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
    if cols == 9: columns = columns + ['fragment_shape(true)']
    df.columns = columns
    df['fragment_shape'] = df['fragment_shape'].str.replace('NULL','-1')
    if cols == 8: df['fragment_shape(true)'] = df['fragment_shape']
    df_train = df.sample(frac=train_frac, random_state=1)
    df_validation = pd.merge(df,df_train,how='outer',indicator=True,on=['tx','start','end']).query('_merge=="left_only"').drop(['_merge'],axis=1)
    csv_train = fragment.replace('.txt', '.train_truenull.txt')
    csv_validate = fragment.replace('.txt', '.validation_truenull.txt')
    df_train.to_csv(csv_train, header=None, sep='\t', index=False)
    df_validation.iloc[:,0:9].to_csv(csv_validate, header=None, sep='\t', index=False)
    
    return csv_train,csv_validate
    
def shuffle_data_sequence_shape(data, mode='Sequence', seed=1234):
    np.random.seed(seed)
    savefn = data.replace('.txt', '.shuffle{}.txt'.format(mode))
    SAVEFN = open(savefn, 'w')
    
    base_choice_dict = {'A':['T','C','G'], 'T':['A', 'C', 'G'], 'C':['A','T', 'G'], 'G':['A','T', 'C']}
    
    with open(data, 'r') as DATA:
        for line in DATA:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            seq = list(arr[6])
            shape = arr[7].split(',')
            if mode == 'Sequence':
                np.random.shuffle(seq)
                seq_shuffle = ''.join(seq)
                arr[6] = seq_shuffle
            if mode == 'Base':
                seq_shuffle = ''.join([np.random.choice(base_choice_dict[b], 1)[0] if float(v) == -1 else b for b,v in zip(seq, shape)])
                arr[6] = seq_shuffle
            if mode == 'Shape':
                np.random.shuffle(shape)
                shape_shuffle = ','.join(shape)
                arr[7] = shape_shuffle
            if mode == 'Shapevalid':
                valid_shape = [v for b,v in zip(seq, shape) if float(v) != -1]
                np.random.shuffle(valid_shape)
                seq_shuffle_ls = []
                n = 0
                for b,v in zip(seq, shape):
                    if float(v) == -1:
                        seq_shuffle_ls.append('-1')
                    else:
                        seq_shuffle_ls.append(valid_shape[n])
                        n += 1
                shape_shuffle = ','.join(seq_shuffle_ls)
                arr[7] = shape_shuffle
            SAVEFN.write('\t'.join(arr)+'\n')
    SAVEFN.close()
# shuffle_data_sequence_shape('/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt', mode='Sequence', seed=1234)

def fragment_data_split_by_rpkm(fragment, rpkm):
    df_fragment = pd.read_csv(fragment, header=None, sep='\t')
    df_rpkm= pd.read_csv(rpkm, header=None, sep='\t')
    df_rpkm.columns = ['tx', 'length', 'type', 'rpkm', 'null_pct']
    tx_rpkm_dict = {i:float(j) for i,j in zip(df_rpkm['tx'], df_rpkm['rpkm'])}
    df_fragment['rpkm'] = [tx_rpkm_dict[i] for i in df_fragment[0]]
    df_fragment.sort_values(by='rpkm', inplace=True)
    df_fragment.drop(['rpkm'], axis=1, inplace=True)
    entry_num = df_fragment.shape[0]
    low_expr = fragment.replace('.txt', '.lowExp.txt')
    df_fragment.iloc[0:int(entry_num/2.0),:].to_csv(low_expr, header=False,index=False,sep='\t')
    high_expr = fragment.replace('.txt', '.highExp.txt')
    df_fragment.iloc[int(entry_num/2.0):,:].to_csv(high_expr, header=False,index=False,sep='\t')
         
def data_random_null(fragment, null_pct=0.1, col=8, savefn=None):
    df = pd.read_csv(fragment, header=None, sep='\t')
    print(df.head())
    if col == 8:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
        print(columns)
        df.columns = columns
        df['fragment_shape(true)'] = df['fragment_shape']
        columns = columns+['fragment_shape(true)']
    if col == 9:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
        df.columns = columns
    
    validation_shape = np.array([i.split(',') for i in df['fragment_shape']])
    random_null = random_mask(x=validation_shape.shape[0],
                              y=validation_shape.shape[1],
                              mask_num=int(np.product(validation_shape.shape)*null_pct))
    
    validation_shape[random_null==0] = '-1'
    validation_shape = [','.join(i) for i in validation_shape]
    df['fragment_shape'] = validation_shape
    
    if savefn is None:
        savefn = fragment.replace('.txt','_randomNULL{}.txt'.format(null_pct))
    df[columns].to_csv(savefn, header=None, sep='\t', index=False)

def data_null_even_in_interval(fragment, null_pct=0.1, col=8, savefn=None, seed=1234):
    print('fragment', fragment)
    np.random.seed(seed)
    df = pd.read_csv(fragment, header=None, sep='\t')
    print(df.head())
    if col == 8:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
        print(columns)
        df.columns = columns
        df['fragment_shape(true)'] = df['fragment_shape']
        columns = columns+['fragment_shape(true)']
    if col == 9:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
        df.columns = columns
    
    validation_shape = np.array([list(map(float, i.split(','))) for i in df['fragment_shape']])
    mask_num=int(np.product(validation_shape.shape)*null_pct)
    for i in range(10):
        idx_random_select_idx = get_interval_random_idx(validation_shape, num=int(mask_num/10), interval_min=i/10, interval_max=(i+1)/10, replace=False)
        print(i,i/10,(i+1)/10,len(idx_random_select_idx[0]))
        validation_shape[tuple(idx_random_select_idx)] = -1
    
    validation_shape = [','.join(list(map(str, i))) for i in validation_shape]
    df['fragment_shape'] = validation_shape
    
    if savefn is None:
        savefn = fragment.replace('.txt','_evenrandomNULL{}.txt'.format(null_pct))
    df[columns].to_csv(savefn, header=None, sep='\t', index=False)
    print('write to {}'.format(savefn))
    
def data_random_nullfragament(fragment, null_pct=0.1, col=8, savefn=None, mode='1perfragment', null_len=5, window_len=100, max_null_len=10):
    df = pd.read_csv(fragment, header=None, sep='\t')
    print(df.head())
    if col == 8:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
        print(columns)
        df.columns = columns
        df['fragment_shape(true)'] = df['fragment_shape']
        columns = columns+['fragment_shape(true)']
    if col == 9:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
        df.columns = columns
    if mode == '1perfragment':
        validation_shape_ls = []
        for i in df['fragment_shape']:
            start_idx = np.random.choice(window_len-null_len, 1)[0]
            validation_shape = ['-1' if n>=start_idx and n<start_idx+null_len else v for n,v in enumerate(i.split(','))]
            validation_shape_ls.append(','.join(validation_shape))
        df['fragment_shape'] = validation_shape_ls
    if mode == 'randomNperfragment':
        validation_shape_ls = []
        for i in df['fragment_shape']:
            null_pos = random_n(l=i.split(','), pct=null_pct, max_len=max_null_len)
            validation_shape = ['-1' if n in null_pos else v for n,v in enumerate(i.split(','))]
            validation_shape_ls.append(','.join(validation_shape))
        df['fragment_shape'] = validation_shape_ls
    
    if savefn is not None:
        df[columns].to_csv(savefn, header=None, sep='\t', index=False)

def fragment_random_based_on_dist(fragment, col=8, savefn=None):
    df = pd.read_csv(fragment, header=None, sep='\t')
    if col == 8:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
        print(columns)
        df.columns = columns
        df['fragment_shape(true)'] = df['fragment_shape']
        columns = columns+['fragment_shape(true)']
    if col == 9:
        columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
        df.columns = columns
    validation_shape_ls = []
    for i in df['fragment_shape']:
        validation_shape_ls.append(i.split(','))
        
    ratio_diff_dict,stat_min_set = ratio_set(num_fragment=df.shape[0]) 
    count_ls = ratio_diff_dict[stat_min_set]['count_ls']
    
    null_num_dict = nested_dict({1:count_ls[0], 2:count_ls[1], 3:count_ls[2], 4:count_ls[3], 5:count_ls[4], 6:count_ls[5], 7:count_ls[6], 8:count_ls[7], 9:count_ls[8], 15:count_ls[9]})
    fragment = len(validation_shape_ls)
    fragment_len = len(validation_shape_ls[0])
    for i,j in null_num_dict.items():
        t = 0
        while t < j:
            fragment_random = np.random.choice(fragment, 1)[0]
            s = np.random.choice(fragment_len, 1)[0]
            if i == 15:
                e = s + np.random.choice([10,11,12,13,14,15,16,17,18,19,20], 1)[0]
            else:
                e = s + i
            if e > fragment_len: continue
            if 'NULL' in validation_shape_ls[fragment_random][s:e]:
                continue
            else:
                validation_shape_ls[fragment_random][s:e] = ['-1'] * (e-s)
                t += 1
    df['fragment_shape'] = [','.join(i) for i in validation_shape_ls]
    
    if savefn is not None:
        print('write to: {}'.format(savefn))
        df[columns].to_csv(savefn, header=None, sep='\t', index=False)

def calc_null_ratio(null_pattern_stat_dict=None, num_fragment = 15805, fragment_length = 100, stat_min_count = 1000):
    if null_pattern_stat_dict is None:
        null_pattern_stat_dict = nested_dict({1:6265, 2:2180, 3:1150, 4:892, 5:655, 6:499, 7:432, 8:372, 9:315, 15:15627})
    stat_min = min(null_pattern_stat_dict.values())
    ratio_ls = []
    for i,j in null_pattern_stat_dict.items():
        ratio = j/stat_min
        ratio_ls.append(ratio)
    ratio_sum = sum(ratio_ls)
#     print("ratio over minimal stat:", ratio_ls, ratio_sum)

    fragment_base = num_fragment * fragment_length

    null_count_ls = []
    count_ls = []
    for i,j in null_pattern_stat_dict.items():
        ratio = j/stat_min
        null_count = ratio * stat_min_count * i
        null_count_ls.append(null_count)
        count_ls.append(ratio * stat_min_count)
    null_count_sum = sum(null_count_ls)
    null_count_ratio = null_count_sum / fragment_base
#     print("null ratio over all fragments:", null_count_ls, null_count_sum, null_count_ratio)
    
    return null_count_ratio, null_count_ls, count_ls

def ratio_set(start=100, end=2000, step=50, null_count_ratio_target=0.3, num_fragment=15805):
    ratio_diff_dict = nested_dict(2, list)
    stat_min_set = 0; ratio_diff_min = 1
    for stat_min in range(start, end, step):
        null_count_ratio, null_count_ls, count_ls = calc_null_ratio(stat_min_count = stat_min, num_fragment=num_fragment)
        print(stat_min, null_count_ratio)
        
        ratio_diff = abs(null_count_ratio-null_count_ratio_target)
        
        ratio_diff_dict[stat_min]['null_count_ratio'] = null_count_ratio
        ratio_diff_dict[stat_min]['null_count_ls'] = null_count_ls
        ratio_diff_dict[stat_min]['ratio_diff'] = ratio_diff
        ratio_diff_dict[stat_min]['count_ls'] = list(map(int, count_ls))
    
        if ratio_diff<ratio_diff_min:
            ratio_diff_min = ratio_diff
            stat_min_set = stat_min
    
    return ratio_diff_dict, stat_min_set
        
def flter_null_fragment(fragment=None, col=7, null_pct=0.4):
    SAVEFN = open(fragment.replace('.txt', '.null{}.txt'.format(null_pct)), 'w')
    with open(fragment, 'r') as FILE:
        for line in FILE:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            shape_ls = arr[col].split(',')
            pct = shape_ls.count('-1') / len(shape_ls)
            if pct<= null_pct:
                SAVEFN.write(line+'\n')
    SAVEFN.close()

def data_add_noise(fragment, ratio=2, col=7, seed=1234, savefn=None, noise=0.05):
    np.random.seed(seed)
    df = pd.read_csv(fragment, sep='\t', header=None)
    SAVEFN = open(savefn, 'w')
    for index,rows in df.iterrows():
        row = list(rows)
        i = row[7]
        shape_ls = list(map(float, i.split(',')))
        for n in range(ratio):
            shape_random_ls = []
            for v in shape_ls:
                if v == -1:
                    v_random = -1
                    shape_random_ls.append(v_random)
                    continue
                else:
                    if v - noise < 0:
                        random_low = -v
                    else:
                        random_low = -noise
                    if v + noise > 1:
                        random_high = 1 - v
                    else:
                        random_high = noise
                    v_random = v + np.random.uniform(low=random_low, high=random_high)
                    shape_random_ls.append(v_random)
            shape_random_str = ','.join(list(map(str, shape_random_ls)))
            SAVEFN.write('\t'.join(list(map(str,row[0:col]+[shape_random_str]+row[col+1:])))+'\n')
    SAVEFN.close()
####################################################################

####################################################################
### data file process
def calc_NULL_percentage(out=None, savefn=None, pureID=1):
	SAVEFN = open(savefn, 'w')
	with open(out, 'r') as OUT:
		for line in OUT:
			line = line.strip()
			if not line or line.startswith('#'): continue
			arr = line.split('\t')
			tx_id = arr[0]
			if pureID:
				tx_id = tx_id.split('.')[0]
			length = int(arr[1])
			rpkm = float(arr[2]) if arr[2] != '*' else arr[2]
			reactivity_ls = arr[3:]
			NULL_pct = reactivity_ls.count('NULL') / float(length)
			print>>SAVEFN,'\t'.join(map(str, [tx_id, length, reactivity_ls.count('NULL'), NULL_pct]))

def shape_dict_stat(shape_dict, fa_dict, trans_dict, RNA_type='lncRNA', trim5Len=5, trim3Len=30):
    stat = nested_dict(1, int)
    for i,j in shape_dict.items():
        if RNA_type:
            if RNA_type == 'all':
                pass
            elif trans_dict[i]['type'] != RNA_type:
                continue
        stat['total_bases'] += (int(j['length']) - trim5Len - trim5Len)
        stat['total_bases(NULL)'] += j['reactivity_ls'][trim5Len:-trim3Len].count('NULL')+j['reactivity_ls'][trim5Len:-trim3Len].count('-1.0')++j['reactivity_ls'][trim5Len:-trim3Len].count('-1')
        for v,b in zip(j['reactivity_ls'][trim5Len:-trim3Len], fa_dict[i.split('.')[0]][trim5Len:-trim3Len]):
            if v in ['NULL', '-1.0', '-1']:
                stat[b+'(NULL)'] += 1
            else:
                stat[b+'(valid)'] += 1
            stat[b] += 1
    stat['total_bases(NULL_pct)'] = stat['total_bases(NULL)'] / float(stat['total_bases'])
    for b in ['A', 'T', 'C', 'G']:
        if float(stat['total_bases(NULL)']) == 0: 
            stat[b+'(NULL_pct)'] = 0
        else:
            stat[b+'(NULL_pct)'] = stat[b+'(NULL)'] / float(stat['total_bases(NULL)'])
    return stat

def fragment_stat(fragment=None):
	if fragment is None:
		fragment = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen50.sliding30.txt'
	df = pd.read_csv(fragment, header=None, sep='\t')
	columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape']
	df.columns = columns
	# df['null_pct'] = [i.split(',').count('NULL')/float(len(i.split(','))) for i in df['fragment_shape']]
	df['null_pct(label)'] = pd.cut(x=list(df['null_pct']),bins=[i/float(10) for i in range(0,11)],include_lowest=True)
	df_stat = df.groupby('null_pct(label)').count()['tx']
	print(df_stat,df_stat.sum())
    
def predict_to_shape(validation, predict, shape_out):
    df_validation = pd.read_csv(validation, header=None, sep='\t')
    columns = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validation.columns = columns
    df_predict = pd.read_csv(predict, header=None, sep='\t')
    df_predict.columns = ['fragment_shape(true)']
    df_validation['fragment_shape(true)'] = df_predict['fragment_shape(true)']
    
    shape_predict_dict = nested_dict(2, list)
    for n,(index,row) in enumerate(df_validation.iterrows()):
        if n%10000==0: print('process:',n)
        tx = row['tx']
        shape = list(map(float, row['fragment_shape(true)'].split(',')))
        for n,i in enumerate(range(row['start'], row['end'])):
            shape_predict_dict[tx][i].append(float(shape[n]))
        
    with open(shape_out, 'w') as OUT:
        for tx in shape_predict_dict:
            idx_ls = sorted(shape_predict_dict[tx].keys())
            # value_ls = [np.mean(shape_predict_dict[tx][i]) for i in idx_ls]
            value_ls = []
            for i in idx_ls:
                v = shape_predict_dict[tx][i]
                if len(v) == 1 and v[0]>=0:
                    v_mean = np.mean(v)
                elif len(v) == 1 and v[0]<0:
                    v_mean = -1
                elif len(v) > 1 and -1 in v:
                    v_mean = np.mean([v2 for v2 in v if v2>=0])
                    if np.isnan(v_mean): v_mean = -1
                else:
                    v_mean = np.mean(v)
                if -1 < v_mean < 0:
                    print('unkown list', v)
                    sys.exit()
                if np.isnan(v_mean):
                    print('unkown list', v)
                    sys.exit()
                value_ls.append(v_mean)
            out_ls = [tx, str(len(value_ls)), '*']+value_ls
            out_ls = map(str, out_ls)
            OUT.write('\t'.join(out_ls)+'\n')
            
def sort_two_shape(shape1=None, value_col1=3, shape2=None):
    df1 = pd.read_csv(shape1, header=None, sep='\t')
    df1_cols_origin = df1.columns
    df2 = pd.read_csv(shape2, header=None, sep='\t')
    if df1.shape[0] != df2.shape[0]:
        print('two file not same entry num: n1={}, n2={}'.format(df1.shape[0], df2.shape[0]))
        sys.exit()
    df1['n(NULL)'] = [i.split(',').count('NULL')+i.split(',').count('-1') for i in df1[value_col1]]
    df1.sort_values(by='n(NULL)', inplace=True)
    index_ls = list(df1.index)
    df2_sort = df2.iloc[index_ls, :]
    
    savefn1 = shape1.replace('.shape', '.sort.shape')
    savefn2 = shape2.replace('.shape', '.sort.shape')
    
    df1[df1_cols_origin].to_csv(savefn1, header=False, index=False, sep='\t')
    df2_sort.to_csv(savefn2, header=False, index=False, sep='\t')
    
    return savefn1,savefn2

def replace_predict_at_valid_with_original(prediction_all, shape_matrix, shape_true_matrix): # prediction_all: list of list, shape_matrix: matrix (sample x length)
    prediction_all_new = []
    if len(prediction_all) != shape_matrix.shape[0]:
        print('unconsistent dim between prediction_all() and shape_matrix ()'.format(len(prediction_all), shape_matrix.shape[0]))
        sys.exit()
    for i in range(len(prediction_all)):
        predict = prediction_all[i]
        shape = shape_matrix[i,]
        shape_true = shape_true_matrix[i,]
        predict_new = []
        for p,s,s_t in zip(predict, shape, shape_true):
            if s == -1:
                predict_new.append(p)
            else:
                predict_new.append(float(s_t))
        prediction_all_new.append(predict_new)
    return prediction_all_new

def write_featuremap(featuremap, txt, lstm_output_time):
    featuremap_pt = txt.replace('.txt', '.pt')
    torch.save(featuremap, featuremap_pt)
    
    for n,i in enumerate(featuremap):
        print('featuremap',i)
        txt_n = txt.replace('.txt', '.out{}.txt'.format(n))
        out = i[0][:,:,lstm_output_time]
        with open(txt_n, 'w') as F_N:
            for v in out.cpu().numpy():
                F_N.write(','.join(map(str,v))+'\n')
####################################################################

####################################################################
### matrix, array 
def random_n(l, pct=0.2, max_len=10):
    pos = []
    base = 0
    while base < len(l)*pct:
        s = np.random.choice(len(l), 1)[0]
        e = s + np.random.choice(max_len, 1)[0]
        status = 0
        for i in range(s, e):
            if i not in pos:
                pass
            else:
                status += 1
        if status:
           continue
        else:
            for i in range(s, e):
                pos.append(i)
            base += (e-s)
    # print(pos)
    return pos

def tile(a, dim, n_tile):
    # https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def random_mask(x,y,mask_num=10):
    a = np.ones(x*y)
    a[:mask_num] = 0
    np.random.shuffle(a)
    return a.reshape(x,y)

def encode_rna(seq, feature_size=4):
    if feature_size == 4:
        d = {'A':[1,0,0,0],
             'T':[0,1,0,0],
             'C':[0,0,1,0],
             'G':[0,0,0,1],
             'N':[0,0,0,0]}
    elif feature_size == 6:
        d = {'A':[1,0,0,0,1,0],
             'T':[0,1,0,0,1,0],
             'C':[0,0,1,0,0,1],
             'G':[0,0,0,1,0,1],
             'N':[0,0,0,0,0,0]}
    else:
        print('feature_size({}) not define'.format(feature_size))
        sys.exit()
    seq_feature = np.zeros((len(seq), feature_size))
    for n,i in enumerate(seq):
        seq_feature[n,] = d[i]
    return seq_feature

def get_interval_random_idx(a, num=10, interval_min=10, interval_max=20, replace=False):
    interval_idx = np.where((interval_min <= a) & (a <= interval_max))
    print(num,interval_min,interval_max,len(interval_idx[0]))
    interval_idx_dict = nested_dict(3, list)
    for n,(i,j) in enumerate(zip(interval_idx[0], interval_idx[1])):
        interval_idx_dict[n]['row_idx'] = i
        interval_idx_dict[n]['col_idx'] = j
    idx_random_select = np.random.choice(list(interval_idx_dict.keys()), num, replace=replace)
    idx_random_select_idx = [[], []]
    for n in idx_random_select:
        idx_random_select_idx[0].append(interval_idx_dict[n]['row_idx'])
        idx_random_select_idx[1].append(interval_idx_dict[n]['col_idx'])
    return idx_random_select_idx

def reactivity_ls_null_loc(reactivity_ls, start=5, trim5Len=35):
    s,e=start,start
    pos_ls = []
    prev=reactivity_ls[start]
    if prev == 'NULL':
        e += 1
    for i in range(start+1,len(reactivity_ls)-trim5Len):
        if prev == 'NULL':
            if reactivity_ls[i] == 'NULL':
                e += 1
            else:
                # print s,e
                pos_ls.append([s,e])
                s += 1
        else:
            if reactivity_ls[i] == 'NULL':
                s = i
                e = i+1
            else:
                s += 1
        prev = reactivity_ls[i]
    if prev == 'NULL':
        # print s,e
        pos_ls.append([s,e])
    return pos_ls
####################################################################

####################################################################
### plot functions
def heatmap(df, xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=None, fmt='d', fig_size_x=10, fig_size_y=8):
    fig,ax=plt.subplots(figsize=(fig_size_x,fig_size_y))
    sns.heatmap(df,annot=True,fmt=fmt,linewidths=0.5)
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
def plot_heatmap(fn, savefn, value_col=3, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black'):
    df = pd.read_csv(fn, header=None, sep='\t')
    df['n(NULL)'] = [i.split(',').count('NULL')+i.split(',').count('-1') for i in df[value_col]]
    df.sort_values(by='n(NULL)', inplace=True)
    df_3 = df[value_col].str.split(",",expand=True,)
    df_3.replace('NULL', np.nan, inplace=True)
    df_3.replace('-1', np.nan, inplace=True)
    df_3 = df_3.applymap(lambda x:float(x))

    fig,ax=plt.subplots(figsize=(fig_size_x,fig_size_y))
    g = sns.heatmap(df_3, xticklabels=False, yticklabels=False, vmax=1, vmin=0, cmap=cmap)
    g.set_facecolor(facecolor)
    plt.savefig(savefn)
    plt.close()
    
    return df_3

def sns_color_ls():
    return sns.color_palette("Set1", n_colors=8, desat=.5)*2
    
def cumulate_dist_plot(ls_ls,ls_ls_label,bins=40,title=None,ax=None,savefn=None,xlabel=None,ylabel=None,add_vline=None,add_hline=None,log2transform=0,xlim=None,ylim=None):
    if ax is None:
        with sns.axes_style("ticks"):
            fig,ax = plt.subplots(figsize=(8,8))
            
    color_ls = sns_color_ls()
    
    ls_ls_label = [j+' ('+str(len(i))+')' for i,j in zip(ls_ls,ls_ls_label)]
    
    if log2transform:
        ls_ls = [np.log2(i) for i in ls_ls]
        
    for n,ls in enumerate(ls_ls):
        values,base = np.histogram(ls,bins=bins)
        cumulative = np.cumsum(values)
        cumulative_norm = [i/float(len(ls)) for i in cumulative]
        ax.plot(base[:-1],cumulative_norm,color=color_ls[n],label=ls_ls_label[n])
        print("plot line num: {}".format(n))
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("Accumulate percent over total")
        
    if title is not None:
        ax.set_title(title)
        
    if add_vline is not None:
        for vline in add_vline:
            ax.axvline(vline,ls="--", color='lightgrey')
            
    if add_hline is not None:
        for hline in add_hline:
            ax.axhline(hline,ls="--", color='lightgrey')
            
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        
    ax.legend(loc="best")
    
    if savefn is not None:
        plt.savefig(savefn)
        plt.close()
####################################################################

####################################################################
### AUC
def calc_auc(file_list, df_list, ct_list, access_ls, seq, savefn, accessibility_cutoff=3, reactivity_cutoff=0, base_used_ls='ATCGU', plot_ss_ds=0, title=''):
    file_list = file_list[0:]
    df_list = df_list[0:]
    
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    calc_base_num_list = []
    
    for i,base_used in zip(range(len(df_list)), base_used_ls.split(':')):
        df_list[i]['ss_ds'] = list(map(int, ct_list))
        df_list[i]['base'] = list(seq)
        if not access_ls is None:
            df_list[i]['accessibility'] = access_ls
        clean = df_list[i].dropna()
        if not accessibility_cutoff is None:
            select = clean[clean.accessibility >= accessibility_cutoff]
        else:
            select = clean.loc[:,:]
        select = select[select.reactivity >= reactivity_cutoff]
        select = select[select.base.isin(list(base_used))]
        print(select.shape,select.head())
        
        if plot_ss_ds:
            single_strand = select[select.ss_ds == 1]
            double_strand = select[select.ss_ds == 0]
            print(select.head(5))
            print(select['ss_ds'].value_counts())
            print(single_strand.describe())
            print(double_strand.describe())
            sns.distplot(single_strand.dropna()['reactivity'], norm_hist=True, kde=False, bins=10)
            sns.distplot(double_strand.dropna()['reactivity'], norm_hist=True, kde=False, bins=10)
            plt.title(file_list[i])
            plt.show()
            
        # print(i, select.shape, select.head())
        if select.shape[0] < 1:
            print("no enough base", i, base_used,select.shape[0])
            return None,None,None,None
        fpr, tpr, _ = roc_curve(select['ss_ds'], select['reactivity'])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(auc(fpr, tpr))
        calc_base_num_list.append(select.shape[0])
        
    plt.close()
    plt.figure()
    lw = 2
    with sns.axes_style("ticks"):
        fig,ax=plt.subplots(figsize=(8,8))
        for i in range(len(df_list)):
            print("plot: {}, {}, AUC:{}".format(i, file_list[i], roc_auc_list[i]))
            if i > 10:
                continue
            plt.plot(fpr_list[i],tpr_list[i],lw = lw,label = '%s (area = %0.3f)' %(file_list[i],roc_auc_list[i]))
            
    plt.plot([-0.05, 1.05], [-0.05, 1.05], color='black', lw=lw, linestyle='--',label='Luck')
    ax.axis('square')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('base: %s, access: %s, reactivity: %s'%(base_used, accessibility_cutoff, reactivity_cutoff))
    plt.title(title)
    plt.legend(loc=4)
#     plt.legend(bbox_to_anchor=(1, 1), loc=2)
    if not savefn is None:
        plt.tight_layout()
        plt.savefig(savefn)
    else:
        plt.show()

    return fpr_list, tpr_list, roc_auc_list, calc_base_num_list

####################################################################

####################################################################
### CLIP
def bed_get_fa(bed, species='human', extend=0, write_new_bed=0):
    fa_dict = read_fa(fa=None, species=species, pureID=1)
    fa = bed.replace('.bed', '.e{}.fa'.format(extend))
    FA = open(fa, 'w')
    if write_new_bed:
        new_bed = bed.replace('.bed', '.e{}.bed'.format(extend))
        NEW_BED = open(new_bed, 'w')
    with open(bed,'r') as BED:
        for line in BED:
            line = line.strip()
            if not line or line.startswith('#'): continue
            arr = line.split('\t')
            if arr[0] in fa_dict:
                if extend:
                    arr[1] = int(arr[1]) - extend
                    arr[2] = int(arr[2]) + extend
                    if arr[1] < 0: continue
                    if arr[2] > len(fa_dict[arr[0]][0:]): continue 
                seq = fa_dict[arr[0]][int(arr[1]):int(arr[2])]
                FA.write('>{}:{}-{}'.format(arr[0], arr[1], arr[2])+'\n')
                FA.write(seq+'\n')
                if write_new_bed:
                    NEW_BED.write('\t'.join(list(map(str, arr)))+'\n')
    FA.close()
    if write_new_bed:
        NEW_BED.close()
    return fa

def fimo_convert(fimo_file='/home/gongjing/project/shape_imputation/data/CLIP/test/fimo.txt'):
    RBP_name_dict = read_RBP_name()
    if fimo_file is None:
        fimo_file = '/Share/home/zhangqf7/jinsong_zhang/zebrafish/data/iclip/20181224/Rawdata/shi-zi-5/bwa/CTK_Procedure1/CITS/iCLIP.tag.uniq.clean.CITS.p05.ext20.tranIn.len41.sort.merge.anno.utr3.fimo2/fimo.txt'
    fimo_file_new = fimo_file.replace('.txt', '.new.txt')
    SAVEFN = open(fimo_file_new, 'w')
    with open(fimo_file, 'r') as IN:
        for line in IN:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            arr = line.split('\t')
            name = RBP_name_dict[arr[0]] if arr[0] in RBP_name_dict else arr[0]
            # print(arr)
            start,end = list(map(int, arr[2].split(":")[1].split('-')))
            tx_id = arr[2].split(":")[0]
            end = start + int(arr[4]) 
            start = start + int(arr[3]) - 1
            SAVEFN.write('\t'.join(list(map(str, [tx_id, start, end, name]+arr)))+'\n')
    SAVEFN.close()
    
def bed_fimo(bed, species='human', motif='FXR'):
    bed_fa = bed_get_fa(bed=bed, species=species)
    # search with fimo
    # /home/gongjing/software/meme_4.12.0/bin/fimo -oc ./test --thresh 0.05 --norc ./motif/Collapsed.used.meme ./human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.bed.fa
    motif_meme='/home/gongjing/project/shape_imputation/data/CLIP/motif/Collapsed.used.meme'
    fimo_dir=bed.replace('.bed', '.fimo')
    fimo='/home/gongjing/software/meme_4.12.0/bin/fimo'
    subprocess.call(["{} -oc {} --thresh 0.05 --norc {} {}".format(fimo, fimo_dir, motif_meme, bed_fa)], shell=True)
    fimo_convert('{}/fimo.txt'.format(fimo_dir))
    subprocess.call(['''cd {}; grep "{}" fimo.new.txt > fimo.new.{}.txt'''.format(fimo_dir, motif, motif)], shell=True)

####################################################################

####################################################################
### save & load model
def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pt', '.best.pt'))

def load_state(path, model, optimizer=None):

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))

        if optimizer != None:
            best_val_loss = checkpoint['best_val_loss']
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, epoch))
            return best_val_loss, epoch
    else:
        print("=> no checkpoint found at '{}'".format(path))

def load_module_state_dict(net, state_dict, add=False, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    #print (net)
    own_state = net.state_dict()
    #print(own_state.keys())
    state_dict_new_keys = []
    for name, param in state_dict.items():
        if add:
            name = "module."+name
        else:
            name = name.replace("module.","")
        #print(name)
        state_dict_new_keys.append(name)
        if name in own_state:
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict_new_keys)
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))
####################################################################
        
####################################################################
### custome loss function, LR
class loss_shape_exp(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, beta=2): # x:predict, y:true
        return torch.mean(torch.exp(beta * y) * torch.pow((x-y), 2)) 

class reg_focal_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, gt, pos_neg_cutoff=0.7):
        """ Modified focal loss. Runs faster but costs a little bit more memory
        Arguments:
        pred (batch x c x w x h x l)
        gt   (batch x c x w x h x l)
        """
        pos_inds = gt.eq(pos_neg_cutoff).float()
        neg_inds = gt.lt(pos_neg_cutoff).float()

        neg_weights = torch.pow(1 - gt, 4)

        pred = torch.clamp(pred, 0.01, 0.99)
        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # print(num_pos,pos_loss, neg_loss)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss
    
class smooth_l1_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, beta=1.0 / 9, size_average=True):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 400))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
####################################################################

####################################################################
### from SPOT-RNA: https://github.com/jaswindersingh2/SPOT-RNA/blob/master/utils.py
### not used
def one_hot(seq):
    RNN_seq = seq
    BASES = 'ATCGN'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])

    return feat

def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)

def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)

def get_data(seq):
    seq_len = len(seq)
    one_hot_feat = encode_rna(seq)
    #print(one_hot_feat[-1])
    zero_mask = z_mask(seq_len)[None, :, :, None]
    label_mask = l_mask(one_hot_feat, seq_len)
    temp = one_hot_feat[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    
    return seq_len, [i for i in (feature.astype(float)).flatten()], [i for i in zero_mask.flatten()], [i for i in label_mask.flatten()]
####################################################################
