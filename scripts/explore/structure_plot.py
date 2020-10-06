from __future__ import print_function
import argparse
from nested_dict import nested_dict
import pandas as pd
import numpy as np
from pyfasta import Fasta

import Visual, Structure

def read_dot(dot=None):
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
	print(dot_dict.keys())
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
	print(out_dict.keys())
	return out_dict.to_dict()

def read_fa(fa=None, species='human', pureID=1):
    if fa is None:
        if species == 'mouse':
            fa = '/home/gongjing/project/shape_imputation/data/ref/mm10/mm10_transcriptome.fa'
        if species == 'human':
            fa = '/home/gongjing/project/shape_imputation/data/ref/hg38/hg38_transcriptome.fa'
        if species == 'human_rRNA':
            fa = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/rRNA_human/ribosomalRNA_4.fa'
        if species == 'human_rfam':
            fa = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/Rfam_human/human.dot.dedup.fa'
        if species == 'mouse(CIRSseq)':
            fa = '/home/gongjing/project/shape_imputation/data/CIRSseq/cirs_txid.fa'
    fa_dict1 = Fasta(fa, key_fn=lambda key:key.split("\t")[0])
    if pureID:
        fa_dict = {i.split()[0].split('.')[0]:j[0:] for i,j in fa_dict1.items()}
    else:
        fa_dict = {i.split()[0]:j[0:] for i,j in fa_dict1.items()}
    print(list(fa_dict.keys())[0:3])
    return fa_dict

def read_validate(validate, start, end):
	cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
	df_validate = pd.read_csv(validate, header=None, sep='\t')
	df_validate.columns = cols

	shape_random_dict = nested_dict(1, list)
	for v,s,e in zip(df_validate['fragment_shape'], df_validate['start'], df_validate['end']):
		v_ls = v.split(',')
		for n,i in enumerate(range(s,e)):
			shape_random_dict[i].append(float(v_ls[n]))

	shape_random = [np.mean(j) for i,j in sorted(shape_random_dict.items(), key=lambda kv: float(kv[0]))]
	shape_random = shape_random[start:end]

	shape_null_index = [[n+1,n+1] for n,i in enumerate(shape_random) if i == -1]

	return shape_null_index

def get_structure(shape_out, dot, start, end, species, tx, validate, title, savefn):
	out_dict = read_icshape_out(shape_out)
	dot_dict = read_dot(dot)
	fa_dict = read_fa(fa=None, species=species, pureID=1)
	shape_null_index = read_validate(validate=validate, start=start, end=end)

	VARNA="/Users/gongjing/Downloads/VARNAv3-93-src.jar"
	shape = out_dict[tx]['reactivity_ls'][start:end]
	dot = dot_dict['dotbracket'][start:end]
	seq = fa_dict[tx][start:end]

	visual_cmd = Visual.Plot_RNAStructure_Shape(seq, dot, shape, mode='heatmap', title=title, VARNAProg=VARNA, highlight_region=shape_null_index, correctT=True)
	visual_cmd = visual_cmd.replace('outline=#FFFFFF', 'outline=#000000').replace('fill=00FF00', 'fill=#E6E6E6') # #000000,#9E9E9E
	visual_cmd += ' -o {}'.format(savefn)
	print (visual_cmd)

def get_structures(shape_out, dot, start, end, species, tx, validate, title, savefn):
	out_dict = read_icshape_out(shape_out, pureID=0)
	dot_dict = read_dots(dot)
	fa_dict = read_fa(fa=None, species=species, pureID=0)
	# shape_null_index = read_validate(validate=validate, start=start, end=end)

	VARNA="/Users/gongjing/Downloads/VARNAv3-93-src.jar"
	shape = out_dict[tx]['reactivity_ls'][start:end]
	dot = dot_dict[tx]['dotbracket'][start:end]
	seq = fa_dict[tx][start:end]

	shape = ['-1' if i=='NULL' else '1' for i in shape]
	shape_null_index = [[n+1,n+1] for n,i in enumerate(shape) if i == '-1']

	visual_cmd = Visual.Plot_RNAStructure_Shape(seq, dot, shape, mode='heatmap', title=title, VARNAProg=VARNA, highlight_region=shape_null_index, correctT=True)
	visual_cmd = visual_cmd.replace('outline=#FFFFFF', 'outline=#000000').replace('fill=00FF00', 'fill=#E6E6E6') # #000000,#9E9E9E
	visual_cmd += ' -o {}'.format(savefn)
	print (visual_cmd)

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot structure')
    
    parser.add_argument('--shape_out', type=str, default='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape.b28/prediction.18S0.4.AUC.out')
    parser.add_argument('--dot', type=str, default='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape.b28/human_18S.dot')
    parser.add_argument('--validate', type=str, default='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape.b28/windowLen100.sliding100.fulllength18S.validation_randomNULL0.4.txt')
    parser.add_argument('--start', type=int, default=215, help='Plot start index')
    parser.add_argument('--end', type=int, default=310, help='Plot end index')
    parser.add_argument('--species', type=str, default='human_rRNA', help='Species')
    parser.add_argument('--tx', type=str, default='18S', help='Transcript id')
    parser.add_argument('--title', type=str, default='Plot structure title', help='Title of the plot')
    parser.add_argument('--savefn', type=str, default='/Users/gongjing/SeafileSyn/Project/DL_RNA/structure_plot/plot.pdf', help='Path to savefn')
    parser.add_argument('--plot_type', type=str, default='single_tx|shape_dot_pair', help='Plot type')
    
    # get args
    args = parser.parse_args()

    if args.plot_type == 'single_tx':
    	get_structure(shape_out=args.shape_out, dot=args.dot, start=args.start, end=args.end, species=args.species, tx=args.tx, validate=args.validate, title=args.title, savefn=args.savefn)

    for i in [0.1,0.2,0.3,0.4,0.5]:
    	shape_out = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape.b28/prediction.18S{}.AUC.out'.format(i)
    	validate = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape.b28/windowLen100.sliding100.fulllength18S.validation_randomNULL{}.txt'.format(i)
    	title = 'predict: null({}), {}-{}'.format(i, args.start, args.end)
    	savefn = '/Users/gongjing/SeafileSyn/Project/DL_RNA/structure_plot/plot.predict.null.{}.png'.format(i)
    	get_structure(shape_out=shape_out, dot=args.dot, start=args.start, end=args.end, species=args.species, tx=args.tx, validate=validate, title=title, savefn=savefn)

    	shape_ref = '/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape.b28/shape.c200T2M0m0.out'
    	savefn = '/Users/gongjing/SeafileSyn/Project/DL_RNA/structure_plot/plot.ref.null.{}.png'.format(i)
    	title = 'ref: null({}), {}-{}'.format(i, args.start, args.end)
    	get_structure(shape_out=shape_ref, dot=args.dot, start=args.start, end=args.end, species=args.species, tx=args.tx, validate=validate, title=title, savefn=savefn)
    
    # if args.plot_type == 'shape_dot_pair':
    # 	out_dict = read_icshape_out(args.shape_out, pureID=0)
    # 	dot_dict = read_dots(args.dot)
    # 	for i in out_dict:
    # 		title = '{}, len={}'.format(i, out_dict[i]['length'])
    # 		print(out_dict[i]['length'], len(dot_dict[i]['dotbracket']))
    # 		savefn = '/Users/gongjing/SeafileSyn/Project/DL_RNA/structure_plot/Rfam/plot.{}.eps'.format(i.replace('/','_'))
    # 		get_structures(shape_out=args.shape_out, dot=args.dot, start=0, end=out_dict[i]['length'], species='human_rfam', tx=i, validate=args.validate, title=title, savefn=savefn)

if __name__ == '__main__':
    main()


'''
python structure_plot.py --shape_out /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rfam/3.shape/shape.c200T2M0m0.out --dot /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/Rfam_human/human.dot --plot_type shape_dot_pair --species human_rfam|grep ^java > structure_plot.Rfam.sh
'''