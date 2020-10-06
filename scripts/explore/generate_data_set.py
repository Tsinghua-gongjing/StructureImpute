import util
import argparse

def generate_windows(out=None,  window_len_ls=None, sliding_ls=None, species=None, all_valid_reactivity=0, null_pct_max=0.9):
	if out is None:
		out='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out'
	if window_len_ls is None:
		window_len_ls = [50,100]
	if species is None:
		species = 'human'
	fa_dict = util.read_fa(species=species)
	save_dir = out+'.windowsHasNull'
	util.check_dir_or_make(save_dir)
	for window_len in window_len_ls:
		for sliding in range(10,window_len+1,10):
			savefn = save_dir+'/'+'windowLen%s.sliding%s.txt'%(window_len, sliding)
			# util.shape_fragmentation(out=out, savefn=savefn, window_len=window_len, sliding=sliding, all_valid_reactivity=1) # no null
			util.shape_fragmentation(out=out, fa_dict=fa_dict, savefn=savefn, window_len=window_len, sliding=sliding, all_valid_reactivity=all_valid_reactivity, null_pct_max=null_pct_max) # has null

# generate_windows(out='/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out', species='mouse')

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Generate fragment shape data from raw shape.out file')
    
    parser.add_argument('--out', type=str, help='Path to shape.out file')
    parser.add_argument('--species', type=str, default='human', help='Species of reference')
    parser.add_argument('--window_len_ls', type=str, default='50,100', help='Window of fragment')
    parser.add_argument('--all_valid_reactivity', type=int, default=0, help='Whether fragment all base has valid shape(=1) or not (=0)')
    parser.add_argument('--null_pct_max', type=float, default=0.9, help='Max null percentage of fragment')
    
    # get args
    args = parser.parse_args()
    generate_windows(out=args.out, species=args.species, window_len_ls=list(map(int, args.window_len_ls.split(','))), all_valid_reactivity=args.all_valid_reactivity, null_pct_max=args.null_pct_max)

if __name__ == '__main__':
    main()
    
'''
python generate_data_set.py --out /home/gongjing/project/shape_imputation/data/CIRSseq/GSE54106_CIRS-seq_Reactivity_combined.out --species "mouse(CIRSseq)" --all_valid_reactivity 1
'''