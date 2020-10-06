# -*- coding: utf-8 -*-

import util
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "helvetica"
import sys, os
from nested_dict import nested_dict
import numpy as np, pandas as pd
import scipy

def read_icshape_out(out=None, pureID=1):
	# gj.printFuncRun('read_icshape_out')
	# gj.printFuncArgs()
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
	# gj.printFuncRun('read_icshape_out')
	return out_dict.to_dict()

def bed_meta(bed_ls, bed_label_ls, icshape_out, savefn, extend=20):
	out_dict = read_icshape_out(icshape_out)
	bed_mean_dict = nested_dict(1, list)
	for bed in bed_ls:
		with open(bed, 'r') as BED:
			for line in BED:
				line = line.strip()
				if not line or line.startswith('#'): continue
				arr = line.split('\t')
				if not out_dict.has_key(arr[0]):
					continue
				reactivity_ls = (out_dict[arr[0]]['reactivity_ls']+[np.nan]*200)[int(arr[1])-extend:int(arr[2])+extend]
				reactivity_ls = [np.nan if i == "NULL" else i for i in reactivity_ls]
				bed_mean_dict[bed].append(map(float, reactivity_ls))

	fig,ax=plt.subplots(figsize=(12,8))
	for bed,bed_label in zip(bed_ls, bed_label_ls):
		# region_mean = np.array(bed_mean_dict[bed]).mean(axis=0)
		region_mean = list(pd.DataFrame(bed_mean_dict[bed]).mean())
		print(region_mean)
		ax.plot(region_mean, label="%s(n=%s)"%(bed_label,str(len(bed_mean_dict[bed]))), marker='.')
	plt.axvspan(extend, len(region_mean)-extend-1, color='grey', alpha=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(savefn)
	plt.close()

def bed_meta2(bed_ls, bed_label_ls, icshape_out_ls, savefn, extend=20, species='human'):
    fa_dict = util.read_fa(fa=None, species=species, pureID=1)
    fig,ax=plt.subplots(figsize=(16,8))
    for bed, bed_label, icshape_out in zip(bed_ls, bed_label_ls, icshape_out_ls):
        print(bed, bed_label, icshape_out)
        region_mean,entry_num, df = single_bed_meta(bed, icshape_out, extend, savefn, bed_label, fa_dict)
        print(region_mean)
        print(df.head())
        ax.plot(region_mean, label="%s(n=%s)"%(bed_label,str(entry_num)), marker='.')
#     plt.axvspan(extend, len(region_mean)-extend-1, color='grey', alpha=0.5)
    ax.set_ylim(0,0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()

def bed_meta3(bed_ls, bed_label_ls, icshape_out_ls, savefn, extend=20):
	fig,ax=plt.subplots(figsize=(12,8))

	region_ls_ls = []
	for bed, bed_label, icshape_out in zip(bed_ls, bed_label_ls, icshape_out_ls):
		print(bed, bed_label, icshape_out)
		region_mean,entry_num, df = single_bed_meta(bed, icshape_out, extend)
		region_ls_ls.append(list(df.index))
	region_ls_common = gj.ls_ls_common(region_ls_ls,return_ls=1)
	for bed, bed_label, icshape_out in zip(bed_ls, bed_label_ls, icshape_out_ls):
		print(bed, bed_label, icshape_out)
		region_mean,entry_num, df = single_bed_meta(bed, icshape_out, extend)
		df = df[df.index.isin(region_ls_common)]
		region_mean = df.mean()
		entry_num = df.shape[0]
		ax.plot(region_mean, label="%s(n=%s)"%(bed_label,str(entry_num)), marker='.')
	plt.axvspan(extend, len(region_mean)-extend-1, color='grey', alpha=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(savefn)
	plt.close()

def single_bed_meta(bed, icshape_out, extend=20, savefn=None, bed_label=None, fa_dict=None, cmap='RdYlGn_r', facecolor='#CBCBCB'):
    out_dict = read_icshape_out(icshape_out)
    bed_mean_dict = nested_dict(1, list)
    region_ls = []
    FA = open(bed+'.fa', 'w')
    SHAPE = open(bed+'.shape', 'w')
    SHAPE100 = open(bed+'.shape100.txt', 'w')
    with open(bed, 'r') as BED:
        for line in BED:
            line = line.strip()
            if not line or line.startswith('#'): continue
            arr = line.split('\t')
            if arr[0] not in out_dict:
                continue
            seq = fa_dict[arr[0]][int(arr[1])-extend:int(arr[2])+extend]
            if len(seq) != int(arr[2])+extend-int(arr[1])+extend: continue
            seq100 = fa_dict[arr[0]][int(arr[2])-50:int(arr[2])+50]
            if len(seq100) != 100: continue
            FA.write('>{}'.format(line)+'\n')
            FA.write(seq+'\n')
            reactivity_ls = (out_dict[arr[0]]['reactivity_ls']+['NULL']*200)[int(arr[1])-extend:int(arr[2])+extend]
            SHAPE.write(line+'\t'+','.join(reactivity_ls)+'\n')
            reactivity_ls = [np.nan if i == "NULL" else i for i in reactivity_ls]
            bed_mean_dict[bed].append(map(float, reactivity_ls))
            region_ls.append('|'.join(arr[0:3]))
            
            reactivity_ls100 = (out_dict[arr[0]]['reactivity_ls']+['NULL']*200)[int(arr[2])-50:int(arr[2])+50]
            reactivity_ls100_str = ','.join(reactivity_ls100).replace('NULL','-1')
            SHAPE100.write('\t'.join(list(map(str, [arr[0], len(fa_dict[arr[0]][0:]), int(arr[2])-50, int(arr[2])+50, '.', '.', seq100, reactivity_ls100_str, reactivity_ls100_str])))+'\n')
    FA.close()
    SHAPE.close()
    SHAPE100.close()
    df = pd.DataFrame(bed_mean_dict[bed])
    df.index = region_ls
    
    if savefn is not None:
        savefn = savefn.replace('.pdf', '.{}.pdf'.format(bed_label))
        fig,ax=plt.subplots(figsize=(10,20))
        print(df.head())
        g = sns.heatmap(df, xticklabels=False, yticklabels=False, cmap=cmap)
        g.set_facecolor(facecolor)
        plt.savefig(savefn)
        plt.close()
        
    return list(df.mean()),len(bed_mean_dict[bed]),df

def two_bed_diff(bed_ls, bed_label_ls, icshape_out_ls, savefn, extend=20, test_significance=1):
	fig,ax=plt.subplots(figsize=(12,8))
	region_mean_ls = []
	region_df_ls = []
	for bed, bed_label, icshape_out in zip(bed_ls, bed_label_ls, icshape_out_ls):
		region_mean,entry_num,region_df = single_bed_meta(bed, icshape_out, extend)
		print(region_mean)
		region_mean_ls.append(region_mean)
		region_df_ls.append(region_df)
		ax.plot(region_mean, label="%s(n=%s)"%(bed_label,str(entry_num)), marker='.')
	region_mean_diff = [i-j for i,j in zip(region_mean_ls[0], region_mean_ls[1])]
	# ax.plot(region_mean_diff, label="diff", marker='.')
	plt.axvspan(extend, len(region_mean)-extend-1, color='grey', alpha=0.5)

	if test_significance:
		pvalue_ls = []
		for i in region_df_ls[0].columns:
			p=scipy.stats.ttest_ind(list_remove_na(region_df_ls[0].loc[:, i]), list_remove_na(region_df_ls[1].loc[:, i]))[1]
			pvalue_ls.append(p)
		print("pvalue:", pvalue_ls)
		df_mean_max = pd.DataFrame(region_mean_ls).max().T
		for n,(i,j) in enumerate(zip(pvalue_ls, df_mean_max.values)):
			if i <=0.01:
				ax.text(n, j+0.01, "**", va='center', ha='center', rotation=0)
			elif i <=0.05:
				ax.text(n, j+0.01, "*", va='center', ha='center', rotation=0)
			else:
				pass

	ax.set_ylim(0,0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(savefn)
	plt.close()

	# region_df.to_csv(savefn.replace('.pdf', '.value.txt'), header=True, index=True, sep='\t')
	with open(savefn.replace('.pdf', '.value.txt'), 'w') as SAVEFN:
		print >>SAVEFN, '\t'.join(map(str, region_mean_diff))
	return region_mean_diff

def list_remove_na(ls):
	return [i for i in ls if not np.isnan(i)]

def two_bed_diff2(bed_ls, bed_label_ls, icshape_out_ls, savefn, extend=20):
	### 先求每个bed所有region的平均，再对这个平均相减取差异
	fig,ax=plt.subplots(figsize=(12,8))
	
	region_mean1,entry_num1,region_df1 = single_bed_meta(bed_ls[0], icshape_out_ls[0], extend)
	region_mean2,entry_num2,region_df2 = single_bed_meta(bed_ls[1], icshape_out_ls[1], extend)
	# region_mean3,entry_num3,region_df3 = single_bed_meta(bed_ls[2], icshape_out_ls[2], extend)
	# region_mean4,entry_num4,region_df4 = single_bed_meta(bed_ls[3], icshape_out_ls[3], extend)
	region_mean_diff = [i-j for i,j in zip(region_mean1, region_mean2)]
	ax.plot(region_mean_diff, label="diff1", marker='.')
	# region_mean_diff = [i-j for i,j in zip(region_mean3, region_mean4)]
	# ax.plot(region_mean_diff, label="diff2", marker='.')

	# region_mean_ls = []
	# for bed, bed_label, icshape_out in zip(bed_ls[0:2], bed_label_ls[0:2], icshape_out_ls[0:2]):
	# 	region_mean,entry_num,region_df = single_bed_meta(bed, icshape_out, extend)
	# 	region_mean_ls.append(region_mean)
	# region_mean_diff = [i-j for i,j in zip(region_mean_ls[0], region_mean_ls[1])]
	# ax.plot(region_mean_diff, label="diff1(%s)"%(entry_num), marker='.')

	# for bed, bed_label, icshape_out in zip(bed_ls[2:4], bed_label_ls[2:4], icshape_out_ls[2:4]):
	# 	region_mean,entry_num,region_df = single_bed_meta(bed, icshape_out, extend)
	# 	region_mean_ls.append(region_mean)
	# region_mean_diff = [i-j for i,j in zip(region_mean_ls[2], region_mean_ls[3])]
	# ax.plot(region_mean_diff, label="diff2(%s)"%(entry_num), marker='.')

	plt.axvspan(extend, len(region_mean1)-extend-1, color='grey', alpha=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(savefn)
	plt.close()

def two_bed_diff3(bed_ls, bed_label_ls, icshape_out_ls, savefn, extend=20):
	### 每个region，求得差异之后再取平均
	fig,ax=plt.subplots(figsize=(12,8))

	region_mean1,entry_num1,region_df1 = single_bed_meta(bed_ls[0], icshape_out_ls[0], extend)
	region_mean2,entry_num2,region_df2 = single_bed_meta(bed_ls[1], icshape_out_ls[1], extend)
	region_mean3,entry_num3,region_df3 = single_bed_meta(bed_ls[2], icshape_out_ls[2], extend)
	region_mean4,entry_num4,region_df4 = single_bed_meta(bed_ls[3], icshape_out_ls[3], extend)
	minus_df1 = region_df1-region_df2
	minus_df1.dropna(how='any', inplace=True)
	minus_df2 = region_df3-region_df4
	minus_df2.dropna(how='any', inplace=True)

	ax.plot(list(minus_df1.mean()), label="diff1(%s)"%(minus_df1.shape[0]), marker='.')
	ax.plot(list(minus_df2.mean()), label="diff2(%s)"%(minus_df2.shape[0]), marker='.')

	plt.axvspan(extend, len(list(minus_df1.mean()))-extend-1, color='grey', alpha=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(savefn)
	plt.close()

if __name__ == '__main__':
	bed_meta2(bed_ls=sys.argv[1].split(':'), bed_label_ls=sys.argv[2].split(':'), icshape_out_ls=sys.argv[3].split(':'), savefn=sys.argv[4], extend=int(sys.argv[5]))
# 	two_bed_diff2(bed_ls=sys.argv[1].split(':'), bed_label_ls=sys.argv[2].split(':'), icshape_out_ls=sys.argv[3].split(':'), savefn=sys.argv[4], extend=int(sys.argv[5]))

'''
# K562
python bed_motif_structure_meta.py /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_valid.bed:/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_valid.bed K562vivo1:K562vivo2 /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out /data/gongjing/project/shape_imputation/results/dmsseq_k562_m6A_e20.pdf 20

python bed_motif_structure_meta.py /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_null.bed:/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1.tx_has_shape_base_null.bed K562vivo1:K562vivo2 /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out /data/gongjing/project/shape_imputation/results/dmsseq_k562_m6A_null_e20.pdf 20

# fibroblast
python bed_motif_structure_meta.py /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1fibroblast.tx_has_shape_base_valid.bed:/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1fibroblast.tx_has_shape_base_valid.bed e1fibroblastvivo1:e1fibroblastvivo2 /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out /data/gongjing/project/shape_imputation/results/dmsseq_e1fibroblast_m6A_valid_e20.pdf 20

python bed_motif_structure_meta.py /home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1fibroblast.tx_has_shape_base_null.bed:/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.e1fibroblast.tx_has_shape_base_null.bed e1fibroblastvivo1:e1fibroblastvivo2 /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out /data/gongjing/project/shape_imputation/results/dmsseq_e1fibroblast_m6A_null_e20.pdf 20
'''