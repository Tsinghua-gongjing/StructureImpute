import pandas as pd
import numpy as np
import util

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
from nested_dict import nested_dict

def known_structure_compare(dot=None, validate=None, predict=None, tx=None, start=0, savefn=None, title='', predict_bases='ATCG', validate_bases='ATCG', save_shapes=1):
    if dot is None: dot = '/home/gongjing/project/shape_imputation/data/Known_Structures/human_small.dot' 
    if validate is None: validate = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation_randomNULL0.1.txt' 
    if predict is None: predict = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation_randomNULL0.1.prediction_trainHasNull_lossAll.txt' 
    if tx is None: tx = '18S' 
    
    dot_dict = util.read_dot(dot)
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validate = pd.read_csv(validate, header=None, sep='\t')
    df_validate.columns = cols
    df_predict = pd.read_csv(predict, header=None, sep='\t')
    df_validate['fragment_shape(predict)'] = df_predict[0]
    df_validate = df_validate[df_validate['tx']==tx]
    # print(df_validate.shape, df_validate.head())
    
    shape_true_dict = nested_dict(1, list)
    for v,s,e in zip(df_validate['fragment_shape(true)'], df_validate['start'], df_validate['end']):
        v_ls = v.split(',')
        for n,i in enumerate(range(s,e)):
            shape_true_dict[i].append(float(v_ls[n]))
    shape_predict_dict = nested_dict(1,list)
    for v,s,e in zip(df_validate['fragment_shape(predict)'], df_validate['start'], df_validate['end']):
        v_ls = v.split(',')
        for n,i in enumerate(range(s,e)):
            shape_predict_dict[i].append(float(v_ls[n]))
    shape_true = [np.mean(j) for i,j in sorted(shape_true_dict.items(), key=lambda kv: float(kv[0]))]
    shape_predict = [np.mean(j) for i,j in sorted(shape_predict_dict.items(), key=lambda kv: float(kv[0]))]
    # for i in sorted(shape_true_dict.keys()):
        # print(i,shape_true_dict[i],shape_predict_dict[i])
            
    # shape_true = [float(j) for i in df_validate['fragment_shape(true)'] for j in i.split(',')]
    # shape_predict = [float(j) for i in df_validate['fragment_shape(predict)'] for j in i.split(',')]
    
    shape_predict_out = savefn.replace('.pdf', '.out')
    with open(shape_predict_out, 'w') as OUT:
        OUT.write('\t'.join(map(str, [tx, len(shape_predict), '*']+shape_predict))+'\n')
    
    shape_true = [np.nan if i == -1 else float(i) for i in shape_true]
    shape_predict = [np.nan if i == -1 else float(i) for i in shape_predict]
    df_true = pd.DataFrame({'reactivity':shape_true})
    df_predict = pd.DataFrame({'reactivity':shape_predict})
    
    tx_dot = list(dot_dict['dotbracket'].replace('.','1').replace('(','0').replace(')','0').replace('>','0').replace('<','0').replace('[','0').replace(']','0').replace('{','0').replace('}','0'))
    
    # # use human_small.dot, origin: 1870 need convert
    # print(len(tx_dot), df_true.shape[0])
    # if len(tx_dot) != df_true.shape[0]: 
    #     print(len(tx_dot), df_true.shape[0])
    #     tx_dot = tx_dot[0:966] + tx_dot[967:]
    #     print(len(tx_dot))
        
    
    file_list = ['True', 'Predict']
    df_list = [df_true, df_predict]

    # savefn = predict.replace('.txt','.auc.pdf')
    # a_ls = util.read_18S_accessibility() # not use access info here: not performace better for imputation
    # print(a_ls, len(a_ls))
    df_true.to_csv('/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/rRNA_18S_true.txt', header=True, sep='\t', index=True)
    fpr_list, tpr_list, roc_auc_list, calc_base_num_list = util.calc_auc(file_list, df_list, tx_dot[0+start:df_true.shape[0]+start], None, dot_dict['seq'][0+start:df_true.shape[0]+start], savefn, None, 0, validate_bases+':'+predict_bases, 0, title)
    # fpr_list, tpr_list, roc_auc_list, calc_base_num_list = util.calc_auc(file_list, df_list, tx_dot[0+start:df_true.shape[0]+start], a_ls, dot_dict['seq'][0+start:df_true.shape[0]+start], savefn, 3, 0, 'AC', 0, title)
    
    shape_random_dict = nested_dict(1, list)
    for v,s,e in zip(df_validate['fragment_shape'], df_validate['start'], df_validate['end']):
        v_ls = v.split(',')
        for n,i in enumerate(range(s,e)):
            shape_random_dict[i].append(float(v_ls[n]))
    shape_random = [np.mean(j) for i,j in sorted(shape_random_dict.items(), key=lambda kv: float(kv[0]))]
    shape_random = [np.nan if i == -1 else float(i) for i in shape_random]
    
    fig,ax=plt.subplots(2,1,figsize=(30,6), sharex=True, sharey=True)
    ax[0].plot(shape_true, color='red', lw=0.5)
    ax[0].set_ylabel('true')
    plt.text(0.98, 0.5, 'AUC=\n{0:.3f}'.format(roc_auc_list[0]), ha='center',va='center', transform=ax[0].transAxes)
    ax[1].plot(shape_predict, color='blue', lw=0.5)
    ax[1].set_ylabel('predict')
    plt.text(0.98, 0.5, 'AUC=\n{0:.3f}'.format(roc_auc_list[1]), ha='center',va='center', transform=ax[1].transAxes)
    for m,i in enumerate(shape_random):
            if np.isnan(i):
                x1 = m-0.1
                x2 = m+0.1
                ax[1].axvspan(x1, x2, color='lightgray', alpha=0.3)
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf', '.track.pdf'))
    plt.close()
    
    if save_shapes:
        df_shapes = pd.DataFrame.from_dict({'True':shape_true, 'Predict':shape_predict, 'Random':shape_random})
        df_shapes.to_csv(savefn.replace('.pdf', '.shape_reactivity.txt'), header=True, index=False, sep='\t')
    
# known_structure_compare()
# known_structure_compare(dot='/home/gongjing/project/shape_imputation/data/Known_Structures/human_28S.dot',tx='28S', start=157) # 前157nt不是28s的

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot AUC for single known structure')
    
    parser.add_argument('--dot', type=str, help='Dot file for known structure')
    parser.add_argument('--validate', type=str, help='Validate fragment file')
    parser.add_argument('--predict', type=str, help='Predicted fragment file')
    parser.add_argument('--tx', type=str, help='Transcript to plot')
    parser.add_argument('--start', type=int, default=0, metavar='N',help='Dot start index')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot')
    parser.add_argument('--title', type=str, default='', help='Title of the plot')
    parser.add_argument('--predict_bases', type=str, default='ATCG', help='Bases considered while calc AUC for predict sample')
    parser.add_argument('--validate_bases', type=str, default='ATCG', help='Bases considered while calc AUC for validate sample')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    known_structure_compare(dot=args.dot, validate=args.validate, predict=args.predict, tx=args.tx, start=args.start, savefn=args.savefn, title=args.title, predict_bases=args.predict_bases, validate_bases=args.validate_bases)
    

if __name__ == '__main__':
    main()






