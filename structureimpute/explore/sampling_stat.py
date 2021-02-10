from nested_dict import nested_dict
import os
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"

def get_stat(sample_num_ls=None):
	if sample_num_ls is None:
		sample_num_ls = [10,20,30,40,50,60,70]
	stat_dict = nested_dict(2, int)
	for i in sample_num_ls:
		path_dir = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_%s/3.shape'%(i)
		print i 

		files = os.listdir(path_dir)
		files = [f for f in files if f.endswith('0m0.out')]
		for file in files:
			path_file = path_dir+'/'+file
			n = 0
			with open(path_file, 'r') as FILE:
				for line in FILE:
					n += 1
			cutoff = file.split('.')[1]
			stat_dict[i*2][cutoff] = n
	stat_df = pd.DataFrame.from_dict(stat_dict, orient='index')
	print stat_df

	fig,ax=plt.subplots()
	ax.plot(stat_df.index, stat_df['c200T2M0m0'], label='BD200RT2', marker='.')
	ax.plot(stat_df.index, stat_df['c100T2M0m0'], label='BD100RT2', marker='.')
	ax.plot(stat_df.index, stat_df['c200T0M0m0'], label='BD200RT0', marker='.')
	ax.plot(stat_df.index, stat_df['c100T0M0m0'], label='BD100RT0', marker='.')
	savefn = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo/3.shape/sampling.pdf'
	# ax.legend()
	plt.legend(bbox_to_anchor=(1, 1), loc=2)
	plt.xlabel('Sampling reads (x10^6)')
	plt.ylabel('# profiled transcript')
	plt.xticks(stat_df.index, stat_df.index, rotation=0)
	# plt.tight_layout()
	plt.savefig(savefn, bbox_inches='tight')
	plt.close()

get_stat()