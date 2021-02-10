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
from scipy import stats

import argparse
import util

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Helvetica"

from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import re

def generate_mask_region_validate(shape_out, tx, species, mask_start, mask_end, fragment_start, fragment_end, savefn_dir, plot_gradient=1):
    out_dict = util.read_icshape_out(out=shape_out, pureID=1)
    fa_dict = util.read_fa(fa=None, species=species, pureID=1)
    
    shape_true_ls = []
    shape_mask_ls = []
    for i in range(fragment_start, fragment_end):
        r = out_dict[tx]['reactivity_ls'][i]
        if i >= mask_start and i < mask_end:
            r_mask = -1
        else:
            r_mask = r
        shape_true_ls.append(r)
        shape_mask_ls.append(r_mask)
    shape_true_ls = ['-1' if i=='NULL' else i for i in shape_true_ls]
    shape_true_ls = map(str, shape_true_ls)
    shape_mask_ls = ['-1' if i=='NULL' else i for i in shape_mask_ls]
    shape_mask_ls = map(str, shape_mask_ls)
    seq = fa_dict[tx][fragment_start:fragment_end]
    
    savefn = '{}/{}.F{}-{}.M{}-{}.txt'.format(savefn_dir, tx, fragment_start, fragment_end, mask_start, mask_end)
    with open(savefn, 'w') as SAVEFN:
        SAVEFN.write('\t'.join(map(str, [tx, '1869', fragment_start, fragment_end, '.', '.', seq, ','.join(shape_mask_ls), ','.join(shape_true_ls)]))+'\n')
    
    if plot_gradient:
        plot_savefn = savefn.replace('.txt', '.gradient.pdf')
        subprocess.call(["cd /home/gongjing/project/shape_imputation/ShapeImputation/scripts; python gradcam_SHAPEimpute.py --filename_validation {} --plot_savefn {}".format(savefn, plot_savefn)], shell=True)
        
    return savefn

def search_dot_long_ss(args):
    # if dot is None: dot = '/home/gongjing/project/shape_imputation/data/Known_Structures/human_18S.dot'
    dot_dict = util.read_dot(dot=args.dot)
    # print(dot_dict['dotbracket'])
    if args.type == 'ss':
        reg_str = "\.{"+str(args.min_len)+",}"
    elif args.type == 'ds':
        reg_str = "[\(\)]{"+str(args.min_len)+",}"
    else:
        return
    
    for n,match in enumerate(re.finditer(reg_str, dot_dict['dotbracket'])):
        # if n >= 3: continue
        
        start = match.span()[0]
        end = match.span()[1]
        middle = int((start+end)/2)
        
        fragment_start = middle - 50
        fragment_end = middle + 50
        
        if fragment_start < 0: continue
        if fragment_end > len(dot_dict['dotbracket']): continue
        
        print(n, match.start(), match.span(), match.group(), middle, fragment_start, fragment_end)
        
        generate_mask_region_validate(shape_out=args.shape_out, tx=args.tx, species=args.species, mask_start=start, mask_end=end, fragment_start=fragment_start, fragment_end=fragment_end, savefn_dir=args.savefn_dir, plot_gradient=args.plot_gradient)
    
def search_dot_pair_stem(args):
    dot_dict = util.read_dot(dot=args.dot)
    import forgi
    d = dot_dict['dotbracket']
    s='...(((((...[[[.))))).((((((((((.(((((((((.....(((.(((..((...(((....((..........))...)))))......(((......((((..((..((....(((..................((((....(((((((.....))))))).....)))).......((((...((((((....))))))...))))....((((((.......(((((.((((...((((.((((((((....))))))))..)))).)))).....)))))......))))))...........((((.((((......))))))))....)))...))))..))))(((..(.(((....((((((((.......))))))))))).....))))...((((((((....))))...))))))).((((((..........)))))).((((....))))...)))))).).....(.(((...(((((...))))).)))).)).))))))....(((((((((((((....))).))))))).)))......(((.(((.......)))).)).........(((((((((....[[[[.....[[.)))]].......]]]])))))).))))))))))..........(((((.....((((...(((.......(((.(((((((((((((.((((....))))....))))))))..)))))))).......((((.(((((...(((((((......)))))))....)))))))))................................................................................................................................(((((((((..(((((((((..((((((((...(((......)))......))))))))..))....(..((....)))))))))).))))).))))...)))...))))....((((((...((...((((.........))))...))))))))..........[[[[[[.(((..((((((((.(((((....)))))))))))))..)))...[[..))]]...]]]....]]].)))..(((.....((((....))))....)))...]]]..(((((.(((((((..((..(((((((((((((((((....((((........))))........(((((((....(((((........((((((........))))))......)))))...((.((((..(((((((((...(((((((((....)))..((((......))))..)))))).....((((.(((.((((..((((....(((..((((....)))).)))....))))..)))))))..((((((((.....))))))))....))))...)))).)))...).))))))).....)))))))...)).))))))))))...(((((((.....(((.......((..((((....))))..)).....))).....)))))))......(...((((((((........))))))))...).....))))).....((((((((.......))))))))......))...)))))))))).))....((.((.(.((((((((.((.((((((((((((..(((((((((((((((.((((((((((((.....))))))))))))...)))))))))))))))..))))))))))))).)))))))))..).))..))....((((((((((....))))))))))........'
    bg, = forgi.load_rna(s)
    
    pair_dict = nested_dict(2, list)
    
    reg_str = "[\(\)]{"+str(args.min_len)+",}"
    # for n,match in enumerate(re.finditer(reg_str, dot_dict['dotbracket'])):
    for n,match in enumerate(re.finditer(reg_str, s)):
        # if n >= 3: continue
        
        start = match.span()[0] # 0-based
        end = match.span()[1] # # 0-based, not include
        
        start_pair = bg.pairing_partner(start+1) # 1-based
        end_pair = bg.pairing_partner(end) # 1-based
        
        pos_ls = [start, end, start_pair, end_pair]
        # print(pos_ls)
        if None in pos_ls: continue
        middle = int((max(pos_ls) + min(pos_ls))/2)
        mask_len = max(pos_ls) - min(pos_ls)
        if mask_len > 100: continue
        
        fragment_start = middle - 50
        fragment_end = middle + 50
        
        if fragment_start < 0: continue
        if fragment_end > len(dot_dict['dotbracket']): continue
        
        print(n, match.start(), match.span(), match.group(), start_pair, end_pair, middle, fragment_start, fragment_end)
        
        if (end - start) != (abs(start_pair - end_pair)+1): continue
        pair_start = min(start_pair,end_pair)-1
        pair_end = max(start_pair,end_pair)
        
        if end < pair_start:
            gap_start = end
            gap_end = pair_start
        else:
            gap_start = pair_end
            gap_end = start
            
        gap_dot = s[pair_end:start]
        if len(set(gap_dot)) != 1: continue
            
        pair_dict[n]['gap_start'] = gap_start
        pair_dict[n]['gap_end'] = gap_end
            
        pair_dict[n]['stem_start'] = start
        pair_dict[n]['stem_end'] = end
        pair_dict[n]['pair_start'] = pair_start
        pair_dict[n]['pair_end'] = pair_end
        pair_dict[n]['fragment_start'] = fragment_start
        pair_dict[n]['fragment_end'] = fragment_end
        
        generate_mask_region_validate(shape_out=args.shape_out, tx=args.tx, species=args.species, mask_start=start, mask_end=end, fragment_start=fragment_start, fragment_end=fragment_end, savefn_dir='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/ss_pair', plot_gradient=1)
        generate_mask_region_validate(shape_out=args.shape_out, tx=args.tx, species=args.species, mask_start=pair_start, mask_end=pair_end, fragment_start=fragment_start, fragment_end=fragment_end, savefn_dir='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/ss_pair', plot_gradient=1)
        
    pair_df = pd.DataFrame.from_dict(pair_dict, orient='index')
    savefn = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/ss_pair/stem_pair.txt'
    pair_df.to_csv(savefn, header=True, index=True, sep='\t')
    
def plot_pair_stem_gradient(txt=None):
    if txt is None:
        txt = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/ss_pair/stem_pair.txt'
    df_txt = pd.read_csv(txt, header=0, index_col=0, sep='\t')
    print(df_txt)
    
    gradient_dict = nested_dict(3, list)
    
    txt_dir = '/'.join(txt.split('/')[0:-1])
    for index, i in df_txt.iterrows():
        fn = '{}/18S.F{}-{}.M{}-{}.gradient.txt'.format(txt_dir,i['fragment_start'],i['fragment_end'],i['stem_start'],i['stem_end'])
        df = pd.read_csv(fn, header=0, index_col=0, sep='\t')
        print(df)
        
        df_stem = df.iloc[:,i['stem_start']-i['fragment_start']:i['stem_end']-i['fragment_start']]
        df_loop = df.iloc[:,i['gap_start']-i['fragment_start']:i['gap_end']-i['fragment_start']]
        df_pair = df.iloc[:,i['pair_start']-i['fragment_start']:i['pair_end']-i['fragment_start']]
        df_F5 = df.iloc[:,0:(i['stem_end']-i['stem_start'])]
        df_F3 = df.iloc[:,-(i['stem_end']-i['stem_start']):]
        
        df_ls = [df_F5, df_stem, df_loop, df_pair, df_F3]
        cols = ['A-fragment5','B-stem','C-loop','D-pair','E-fragment3']
        for label,ii in zip(cols, df_ls):
            print(label, ii.head())
            seq_mean = ii.loc[['grad(A)', 'grad(U)', 'grad(C)', 'grad(G)'],:].mean(axis=0).mean()
            shape_mean = ii.loc['grad(mask)',:].mean()
            # seq_mean = i.loc[['grad(A)', 'grad(U)', 'grad(C)', 'grad(G)'],:].sum(axis=0).sum()
            # shape_mean = i.loc['grad(mask)',:].sum()
            print(seq_mean, shape_mean)
            
            gradient_dict['seq'][fn][label] = seq_mean
            gradient_dict['shape'][fn][label] = shape_mean
            
    seq_gradient_df = pd.DataFrame.from_dict(gradient_dict['seq'], orient='index')
    shape_gradient_df = pd.DataFrame.from_dict(gradient_dict['shape'], orient='index')
    # print('seq_gradient_df', seq_gradient_df)
    # print('shape_gradient_df', shape_gradient_df)
    
    savefn = txt_dir + '/gradient_seq.dist.pdf'
    fig,ax=plt.subplots()
    # for i in seq_gradient_df.index:
    #     ax.plot(range(0, len(cols)), seq_gradient_df.loc[i, cols], color='grey', alpha=0.3, lw=0.3)
    # df_plot_mean = seq_gradient_df.loc[:, cols].mean(axis=0)
    # ax.plot(range(0, len(cols)), df_plot_mean, color='blue')
    seq_gradient_df.plot(kind='box', ax=ax)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    savefn = txt_dir + '/gradient_shape.dist.pdf'
    fig,ax=plt.subplots()
    # for i in shape_gradient_df.index:
    #     ax.plot(range(0, len(cols)), shape_gradient_df.loc[i, cols], color='grey', alpha=0.3, lw=0.3)
    # df_plot_mean = shape_gradient_df.loc[:, cols].mean(axis=0)
    # ax.plot(range(0, len(cols)), df_plot_mean, color='blue')
    shape_gradient_df.plot(kind='box', ax=ax)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()


def read_gradient_txt(t):
    df = pd.read_csv(t, header=0, index_col=0, sep='\t')
    print(df.head())
    
    return df
        
def plot_dir_gradient_meta(args):
    d = args.savefn_dir
    if d is None: d = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/ss'
    fn_ls = os.listdir(d)
    fn_ls = [i for i in fn_ls if i.endswith('gradient.txt')]
    gradient_dict = nested_dict(3, list)
    for n,fn in enumerate(fn_ls):
        # if n > 1: continue
        
        fragment_start,fragment_end = map(int, fn.split('.')[1].replace('F','').split('-'))
        mask_start,mask_end = map(int, fn.split('.')[2].replace('M','').split('-'))
        df = read_gradient_txt(d+'/'+fn)
        print('process:', fn, fragment_start,fragment_end, mask_start,mask_end)
        
        idx_start = mask_start - fragment_start
        idx_end = mask_end - fragment_start
        mask_len = mask_end - mask_start
        
        df_M = df.iloc[:,idx_start:idx_end]
        df_F5 = df.iloc[:, 0:mask_len]
        df_F3 = df.iloc[:, -mask_len:]
        df_U5 = df.iloc[:,idx_start-mask_len:idx_start]
        df_D3 = df.iloc[:, idx_end:idx_end+mask_len]
        
        df_ls = [df_F5, df_U5, df_M, df_D3, df_F3]
        cols = ['A-fragment5','B-flank5','C-mask','D-flank3','E-fragment3']
        for label,i in zip(cols, df_ls):
            # print(i)
            seq_mean = i.loc[['grad(A)', 'grad(U)', 'grad(C)', 'grad(G)'],:].mean(axis=0).mean()
            shape_mean = i.loc['grad(mask)',:].mean()
            # seq_mean = i.loc[['grad(A)', 'grad(U)', 'grad(C)', 'grad(G)'],:].sum(axis=0).sum()
            # shape_mean = i.loc['grad(mask)',:].sum()
            print(seq_mean, shape_mean)
            
            gradient_dict['seq'][fn][label] = seq_mean
            gradient_dict['shape'][fn][label] = shape_mean
            
    seq_gradient_df = pd.DataFrame.from_dict(gradient_dict['seq'], orient='index')
    shape_gradient_df = pd.DataFrame.from_dict(gradient_dict['shape'], orient='index')
    print('seq_gradient_df', seq_gradient_df)
    print('shape_gradient_df', shape_gradient_df)
    
    savefn = d + '/gradient_seq.dist.pdf'
    fig,ax=plt.subplots()
    # for i in seq_gradient_df.index:
    #     ax.plot(range(0, len(cols)), seq_gradient_df.loc[i, cols], color='grey', alpha=0.3, lw=0.3)
    # df_plot_mean = seq_gradient_df.loc[:, cols].mean(axis=0)
    # ax.plot(range(0, len(cols)), df_plot_mean, color='blue')
    seq_gradient_df.plot(kind='box', ax=ax)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    savefn = d + '/gradient_shape.dist.pdf'
    fig,ax=plt.subplots()
    # for i in shape_gradient_df.index:
    #     ax.plot(range(0, len(cols)), shape_gradient_df.loc[i, cols], color='grey', alpha=0.3, lw=0.3)
    # df_plot_mean = shape_gradient_df.loc[:, cols].mean(axis=0)
    # ax.plot(range(0, len(cols)), df_plot_mean, color='blue')
    shape_gradient_df.plot(kind='box', ax=ax)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
        
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='generate_mask_region_validate')
    
    parser.add_argument('--shape_out', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out', help='shape file')
    parser.add_argument('--tx', type=str, default='18S', help='tx id')
    parser.add_argument('--species', type=str, default='human_rRNA', help='species')
    parser.add_argument('--mask_start', type=int, default=730, help='mask region:start')
    parser.add_argument('--mask_end', type=int, default=740, help='mask region:end')
    parser.add_argument('--fragment_start', type=int, default=700, help='fragment region:start')
    parser.add_argument('--fragment_end', type=int, default=800, help='fragment region:end')
    parser.add_argument('--savefn_dir', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions', help='path to generated validation file')
    
    parser.add_argument('--dot', type=str, help='dot file of the structure', default='/home/gongjing/project/shape_imputation/data/Known_Structures/human_18S.dot')
    parser.add_argument('--min_len', type=int, default=5, help='mask region: min length')
    parser.add_argument('--type', type=str, default='ss', help='search mask region type')
    parser.add_argument('--plot_gradient', type=int, default=0, help='whether plot gradient heatmap for individual searched mask region')
    
    # get args
    args = parser.parse_args()
    
    # generate_mask_region_validate(shape_out=args.shape_out, tx=args.tx, species=args.species, mask_start=args.mask_start, mask_end=args.mask_end, fragment_start=args.fragment_start, fragment_end=args.fragment_end, savefn_dir=args.savefn_dir)
    # search_dot_long_ss(args)
    # search_dot_pair_stem(args)
    # plot_pair_stem_gradient(txt=None)
    
    plot_dir_gradient_meta(args)
    

if __name__ == '__main__':
    main()
    
'''
python rRNA_18s_validate_mask_specific_regions.py --savefn_dir /home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/ds --type ds

python rRNA_18s_validate_mask_specific_regions.py --savefn_dir /home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/ss --type ss --plot_gradient 1

python rRNA_18s_validate_mask_specific_regions.py --savefn_dir /home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/c94_ss --type ss --plot_gradient 1
python rRNA_18s_validate_mask_specific_regions.py --savefn_dir /home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/mask_specific_regions/c94_ds --type ds --plot_gradient 1
'''
    