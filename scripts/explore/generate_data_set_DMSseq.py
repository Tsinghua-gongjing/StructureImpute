import util
import argparse
import numpy as np
import pandas as pd
import os,subprocess
import re,random

def data_random_null_filterNULL(fragment, null_pct=0.1, col=8, savefn=None, seed=1234):
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
        
    x=validation_shape.shape[0]
    y=validation_shape.shape[1]
    valid_score_num = len(validation_shape[validation_shape!='-1'])
    mask_num=int(valid_score_num*null_pct)
    
    idx_null = np.where(validation_shape == '-1')
    idx_valid = np.where(validation_shape != '-1')
    
    idx_valid_pairs = ['{}-{}'.format(i,j) for i,j in zip(idx_valid[0], idx_valid[1])]
    np.random.seed(seed)
    np.random.shuffle(idx_valid_pairs)
    
    for i in idx_valid_pairs[0:mask_num]:
        i_0,i_1 = map(int, i.split('-'))
        validation_shape[i_0, i_1] = '-1'    
    
    validation_shape = [','.join(i) for i in validation_shape]
    df['fragment_shape'] = validation_shape
    
    if savefn is None:
        savefn = fragment.replace('.txt','_randomNULL{}.txt'.format(null_pct))
    df[columns].to_csv(savefn, header=None, sep='\t', index=False)
    
    return savefn

def shape_fragmentation(out=None, fa_dict=None, savefn=None, window_len=30, sliding=10, all_valid_reactivity=1, null_pct_max=0.9, base_ls='AC'):
    out_dict = util.read_icshape_out(out)
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
                        
                    fragment_shape_base = []
                    for shape,base in zip(fragment_shape, fragment_seq):
                        if base in base_ls:
                            fragment_shape_base.append(shape)
                        
                    mean_reactivity = util.gini(fragment_shape_base,mode='mean_reactivity',null_pct=1)
                    null_pct = (fragment_shape_base.count('NULL')+fragment_shape_base.count('-1.0')) / float(len(fragment_shape_base))
                    # if len(fragment_shape) != window_len: continue
                    if all_valid_reactivity:
                        if any(v in ['NULL', '-1.0'] for v in fragment_shape_base):
                            pass
                        else:
                            SAVEFN.write('\t'.join(map(str, [i, j['length'], ss, ee, mean_reactivity, null_pct, fragment_seq, ','.join(fragment_shape)]))+'\n')
                    else:
                        if null_pct <= null_pct_max:
                            SAVEFN.write('\t'.join(map(str, [i, j['length'], ss, ee, mean_reactivity, null_pct, fragment_seq, ','.join(fragment_shape)]))+'\n')
                    s += sliding

def generate_windows(out=None,  window_len_ls=None, sliding_ls=None, species=None, all_valid_reactivity=0, null_pct_max=0.9, split_train_validate=0, generate_random_null_and_ratio=0):
    if out is None: out='/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out'
    if window_len_ls is None: window_len_ls = [50,100]
    if species is None: species = 'human'
    fa_dict = util.read_fa(species=species)
    save_dir = out+'.windowsHasNull'
    util.check_dir_or_make(save_dir)
    for window_len in window_len_ls:
        for sliding in range(window_len,window_len+1,10):
            savefn = save_dir+'/'+'windowLen%s.sliding%s.txt'%(window_len, sliding)
            # util.shape_fragmentation(out=out, savefn=savefn, window_len=window_len, sliding=sliding, all_valid_reactivity=1) # no null
            shape_fragmentation(out=out, fa_dict=fa_dict, savefn=savefn, window_len=window_len, sliding=sliding, all_valid_reactivity=all_valid_reactivity, null_pct_max=null_pct_max) # has null
            if split_train_validate: 
                np.random.seed(1234)
                csv_train, csv_validate = util.fragment_split(fragment=savefn, train_frac=0.7, cols=8)
                if generate_random_null_and_ratio:
                    data_random_null_filterNULL(csv_train, null_pct=generate_random_null_and_ratio, col=9, savefn=None, seed=1234)
                    data_random_null_filterNULL(csv_validate, null_pct=generate_random_null_and_ratio, col=9, savefn=None, seed=1234)

def data_agumentation(txt=None, times=20, generate_random_null_and_ratio=0.3):
    if txt is None:
        txt = '/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train_truenull.txt'
    DA_dir = '/'.join(txt.split('/')[0:-1])+'/DA/DA{}'.format(times)
    if not os.path.isdir(DA_dir): os.mkdir(DA_dir)
        
    for i in range(times):
        seed = random.randint(1, 100000000)
        savefn = DA_dir + '/S{}.txt'.format(seed)
        random.seed(seed)
        
        data_random_null_filterNULL(txt, null_pct=generate_random_null_and_ratio, col=9, savefn=savefn, seed=seed)
        
    combine_txt = DA_dir+'.txt'
    cmd = 'cat {}/S* > {}'.format(DA_dir, combine_txt)
    subprocess.call([cmd], shell=True)
                    

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Generate fragment shape data from raw shape.out file')
    
    parser.add_argument('--out', type=str, help='Path to shape.out file')
    parser.add_argument('--species', type=str, default='human', help='Species of reference')
    parser.add_argument('--window_len_ls', type=str, default='100', help='Window of fragment')
    parser.add_argument('--all_valid_reactivity', type=int, default=0, help='Whether fragment all base has valid shape(=1) or not (=0)')
    parser.add_argument('--null_pct_max', type=float, default=0.9, help='Max null percentage of fragment')
    parser.add_argument('--split_train_validate', type=int, default=0, help='split(1) or not (0)')
    parser.add_argument('--generate_random_null_and_ratio', type=float, default=0.0, help='generate(>0) or not (0)')
    
    # get args
    args = parser.parse_args()
    # generate_windows(out=args.out, species=args.species, window_len_ls=list(map(int, args.window_len_ls.split(','))), all_valid_reactivity=args.all_valid_reactivity, null_pct_max=args.null_pct_max, split_train_validate=args.split_train_validate, generate_random_null_and_ratio=args.generate_random_null_and_ratio)
    
    
    # data_agumentation()
    # data_agumentation(times=50)
    # data_agumentation(times=70)
    # data_agumentation(times=100)
    
    # data_agumentation(times=50, txt='/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train_truenull.txt')
    # data_agumentation(times=50, txt='/home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train_truenull.txt')
    # data_agumentation(times=10, txt='/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train_truenull.txt')
    
    # data_random_null_filterNULL(fragment='/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt', null_pct=0.3, col=9, savefn=None, seed=1234)
    data_random_null_filterNULL(fragment='/home/gongjing/project/shape_imputation/data/DMSseq_K562_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.txt', null_pct=0.3, col=9, savefn=None, seed=1234)

if __name__ == '__main__':
    main()
    
'''
python generate_data_set.py --out /home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out --species "human" --all_valid_reactivity 1


python generate_data_set.py --out /home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo/3.shape/shape.c200T2M0m0.out --all_valid_reactivity 0 --split_train_validate 1
'''