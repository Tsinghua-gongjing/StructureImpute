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
	return dot_dict.to_dict()

def dot_type(dot):
	dot_dict = read_dot(dot)
	structureinfo = Structure.parse_structure(dot_dict['dotbracket'])

	savefn = dot.replace('.dot', '.dot.txt')
	with open(savefn, 'w') as SAVEFN:
		for i in structureinfo.linking_bases:
			SAVEFN.write('\t'.join(map(str, [i, 'S-linking_bases']))+'\n')
		for i in structureinfo.dangling_bases:
			SAVEFN.write('\t'.join(map(str, [i, 'S-dangling_bases']))+'\n')
		for i in structureinfo.multiloop_bases:
			SAVEFN.write('\t'.join(map(str, [i, 'S-multiloop_bases']))+'\n')
		for i in structureinfo.bulge_bases:
			SAVEFN.write('\t'.join(map(str, [i, 'S-bulge_bases']))+'\n')
		for i in structureinfo.interior_bases:
			SAVEFN.write('\t'.join(map(str, [i, 'S-interior_bases']))+'\n')
		for i in structureinfo.hairpin_bases:
			SAVEFN.write('\t'.join(map(str, [i, 'S-hairpin_bases']))+'\n')
		for i in structureinfo.stacking_middle:
			SAVEFN.write('\t'.join(map(str, [i[0], 'D-stacking_middle']))+'\n')
			SAVEFN.write('\t'.join(map(str, [i[1], 'D-stacking_middle']))+'\n')
		for i in structureinfo.stacking_closing:
			SAVEFN.write('\t'.join(map(str, [i[0], 'D-stacking_closing']))+'\n')
			SAVEFN.write('\t'.join(map(str, [i[1], 'D-stacking_closing']))+'\n')
		for i in structureinfo.mutiloop_closing:
			SAVEFN.write('\t'.join(map(str, [i[0], 'D-mutiloop_closing']))+'\n')
			SAVEFN.write('\t'.join(map(str, [i[1], 'D-mutiloop_closing']))+'\n')
		for i in structureinfo.hairpin_closing:
			SAVEFN.write('\t'.join(map(str, [i[0], 'D-hairpin_closing']))+'\n')
			SAVEFN.write('\t'.join(map(str, [i[1], 'D-hairpin_closing']))+'\n')
		for i in structureinfo.interior_closing:
			SAVEFN.write('\t'.join(map(str, [i[0], 'D-interior_closing']))+'\n')
			SAVEFN.write('\t'.join(map(str, [i[1], 'D-interior_closing']))+'\n')
		for i in structureinfo.pseudoknot_bps:
			SAVEFN.write('\t'.join(map(str, [i[0], 'D-pseudoknot_bps']))+'\n')
			SAVEFN.write('\t'.join(map(str, [i[1], 'D-pseudoknot_bps']))+'\n')

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot structure')
    
    parser.add_argument('--dot', type=str, default='/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/hek_wc_vivo_rRNA/3.shape.b28/human_18S.dot')
    
    # get args
    args = parser.parse_args()
    
    dot_type(dot=args.dot)

if __name__ == '__main__':
    main()