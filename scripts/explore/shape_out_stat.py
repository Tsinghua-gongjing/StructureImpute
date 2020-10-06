import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
import sys, os
from nested_dict import nested_dict
import pandas as pd
import numpy as np
from pyfasta import Fasta
import os
import re
from scipy import stats
import util

def get_stat(coverage_ls=None, RT_ls=None, prefix=None):
	if coverage_ls is None:
		coverage_ls = [0,50,100,150,200,250]
	if RT_ls is None:
		RT_ls = [0,1,2,3]
	if prefix is None:
		prefix = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/3.shape'

	fa_dict = util.read_fa(species='human')
	trans_dict = util.loadTransGtfBed2(species='human')

	stat_dict = nested_dict(3, int)
	for coverage in coverage_ls:
		for RT in RT_ls:
			shape_out = '%s/shape.c%sT%sM0m0.out'%(prefix, coverage, RT)
			out_dict = util.read_icshape_out(shape_out)
			out_dict_stat = util.shape_dict_stat(shape_dict=out_dict, fa_dict=fa_dict, trans_dict=trans_dict, RNA_type='all')
			print(out_dict_stat)

			stat_dict['tx_count'][coverage][RT] = len(out_dict)
			stat_dict['total_bases'][coverage][RT] = out_dict_stat['total_bases']
			stat_dict['total_bases(NULL_pct)'][coverage][RT] = out_dict_stat['total_bases(NULL_pct)']
			stat_dict['A(NULL_pct)'][coverage][RT] = out_dict_stat['A(NULL_pct)']
			stat_dict['T(NULL_pct)'][coverage][RT] = out_dict_stat['T(NULL_pct)']
			stat_dict['C(NULL_pct)'][coverage][RT] = out_dict_stat['C(NULL_pct)']
			stat_dict['G(NULL_pct)'][coverage][RT] = out_dict_stat['G(NULL_pct)']

	print(pd.DataFrame.from_dict(stat_dict['tx_count'], orient='index'))
	print(pd.DataFrame.from_dict(stat_dict['total_bases'], orient='index'))
	print(pd.DataFrame.from_dict(stat_dict['total_bases(NULL_pct)'], orient='index'))

	# fig,ax=plt.subplots()
	# sns.heatmap(tx_count_df,annot=True,fmt='d',linewidths=0.5)
	# ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0)
	# ax.set_xlabel('Cutoff (average RT stop count)')
	# ax.set_ylabel('Cutoff (base density)')
	# savefn=prefix+'/shape.tx_num.pdf'
	# plt.tight_layout()
	# plt.savefig(savefn)
	# plt.close()

	util.heatmap(pd.DataFrame.from_dict(stat_dict['tx_count'], orient='index'), xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=prefix+'/shape.tx_num.pdf', fmt='d')
	util.heatmap(pd.DataFrame.from_dict(stat_dict['total_bases'], orient='index'), xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=prefix+'/shape.total_bases.pdf',fmt='.2g')
	util.heatmap(pd.DataFrame.from_dict(stat_dict['total_bases(NULL_pct)'], orient='index'), xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=prefix+'/shape.total_null_pct.pdf',fmt='.2g')
	util.heatmap(pd.DataFrame.from_dict(stat_dict['A(NULL_pct)'], orient='index'), xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=prefix+'/shape.A_null_pct.pdf',fmt='.2g')
	util.heatmap(pd.DataFrame.from_dict(stat_dict['T(NULL_pct)'], orient='index'), xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=prefix+'/shape.T_null_pct.pdf',fmt='.2g')
	util.heatmap(pd.DataFrame.from_dict(stat_dict['C(NULL_pct)'], orient='index'), xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=prefix+'/shape.C_null_pct.pdf',fmt='.2g')
	util.heatmap(pd.DataFrame.from_dict(stat_dict['G(NULL_pct)'], orient='index'), xlabel='Cutoff (average RT stop count)', ylabel='Cutoff (base density)', savefn=prefix+'/shape.G_null_pct.pdf',fmt='.2g')

get_stat(prefix='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape_different_cutoff')

def null_sequential_pattern(out=None):
	if out is None:
		out = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out'
	out_dict = util.read_icshape_out(out)
	savefn = out.replace('.out', '.null_sequential.txt')
	SAVEFN = open(savefn, 'w')
	for i,j in out_dict.items():
		pos_ls = util.reactivity_ls_null_loc(j['reactivity_ls'])
		for pos in pos_ls:
			SAVEFN.write('\t'.join(map(str, [i, pos[0], pos[1], pos[1]-pos[0]]))+'\n')
	SAVEFN.close()
	df = pd.read_csv(savefn, sep='\t', header=None)
	df.columns = ['tx', 'start', 'end', 'len']
	df['len2'] = [10 if i>=10 else i for i in df['len'] ]

	fig,ax=plt.subplots()
	df_stat = pd.DataFrame(df.groupby('len2').count()['tx'])
	print(df_stat)
	sns.barplot(data=df_stat.T,ax=ax)
	plt.tight_layout()
	plt.savefig(savefn.replace('.txt', '.pdf'))
	plt.close()

# null_sequential_pattern()

def expression_vs_null_pct(out='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out', rep1=None, rep2=None):
	trans_dict = util.loadTransGtfBed2(species='human')
	tx_rpkm_dict = util.read_rpkm_rep(rep1,rep2)
	out_dict = util.read_icshape_out(out)
	savefn = out.replace('.out', '.exp_vs_null.txt')
	SAVEFN = open(savefn, 'w')
	for i,j in out_dict.items():
		if not tx_rpkm_dict.has_key(i): continue
		null_pct = j['reactivity_ls'].count('NULL') / float(j['length'])
		print>>SAVEFN, '\t'.join(map(str, [i, j['length'], trans_dict[i]['type'], tx_rpkm_dict[i], null_pct]))
	SAVEFN.close()

	df = pd.read_csv(savefn, sep='\t', header=None)
	df.columns = ['tx', 'length', 'type', 'rpkm', 'null_pct']
	df['log2(rpkm)']=np.log2(df['rpkm'])

	fig,ax=plt.subplots(figsize=(10,10))
	sns.jointplot(x='log2(rpkm)', y='null_pct', data=df,kind="reg",stat_func=stats.pearsonr)
	plt.tight_layout()
	plt.savefig(savefn.replace('.txt', '.pdf'))
	plt.close()

# expression_vs_null_pct()

def value_dist(out='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out', normalize=True, label=None, savefn=None):
    out_ls = out.split(':')
    label_ls = label.split(':')
    reactivity_stat_df_ls = []
    for out,label in zip(out_ls, label_ls):
        out_dict = util.read_icshape_out(out)
        reactivity_dict = nested_dict(1, list)
        for i,j in out_dict.items():
            value_ls = [float(v) for v in j['reactivity_ls'] if v != 'NULL']
            for v in value_ls:
                reactivity_dict['nonnull(true)'].append(v)
        df = pd.DataFrame({'reactivity':reactivity_dict['nonnull(true)']})
        print('total base', df.shape)
        df['bins'] = pd.cut(df['reactivity'], bins=10)
        count_dict = dict(df['bins'].astype(str).value_counts(normalize=normalize))
        print(count_dict)
        reactivity_stat_df = pd.DataFrame.from_dict(count_dict, orient='index')
        reactivity_stat_df.columns = [label+'\n(n={})'.format(df.shape[0])]
        print(reactivity_stat_df)
        
        reactivity_stat_df_ls.append(reactivity_stat_df)
        
    reactivity_stat_df_all = pd.concat(reactivity_stat_df_ls, axis=1)
    print(reactivity_stat_df_all)
    
    fig,ax=plt.subplots(figsize=(12, 6))
    reactivity_stat_df_all.T.plot(kind='barh', stacked=True, ax=ax)
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.savefig(savefn, bbox_inches='tight')
    plt.close()
    
    return reactivity_stat_df_all

# reactivity_stat_df1 = value_dist('/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out', label='icSHAPE_HEK293:icSHAPE_mES:DMSseq_fibroblast:DMSseq_K562', savefn='/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape_value_dist.pdf')
# reactivity_stat_df2 = value_dist('/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out')


def shape_fragment_null_sequential_pattern(out=None):
    if out is None:
        out = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.randomNullDist.S1234.txt'
    savefn = out.replace('.txt', '.null_sequential.txt')
    SAVEFN = open(savefn, 'w')
    n = 0
    with open(out, 'r') as OUT:
        for line in OUT:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            n += 1
            arr = line.split('\t')
            pos_ls = util.reactivity_ls_null_loc(arr[7].replace('-1', 'NULL').split(','), start=0, trim5Len=0)
            for pos in pos_ls:
                SAVEFN.write('\t'.join(map(str, [arr[0], arr[1], arr[2], arr[3], pos[0], pos[1], pos[1]-pos[0]]))+'\n')
    SAVEFN.close()
    df = pd.read_csv(savefn, sep='\t', header=None)
    df.columns = ['tx', 'tx_len', 'tx_start', 'tx_end', 'start', 'end', 'len']
    null_pct = sum(df['len'])/(n*100)
    print('null_pct', null_pct)
    df['len2'] = [10 if i>=10 else i for i in df['len'] ]

    fig,ax=plt.subplots()
    df_stat = pd.DataFrame(df.groupby('len2').count()['tx'])
    print(df_stat)
    df_stat.sort_index(ascending=False, inplace=True)
    sns.barplot(data=df_stat.T,ax=ax)
    # plt.pie(df_stat.T,labels=df_stat.index)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(savefn.replace('.txt', '.pdf'))
    plt.close()

# shape_fragment_null_sequential_pattern()

# f='/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.train.low_60_1234.null_pattern.x10.chrom0.txt'
# f='/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_x10_then_pct30_maxL20/windowLen100.sliding100.trainx10_randomNULL0.3.txt'
# shape_fragment_null_sequential_pattern(out=f)