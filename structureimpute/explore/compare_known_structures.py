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

def known_structures_compare(dot=None, validate=None, predict=None,savefn=None, start=0, title=''):
    if dot is None: dot = '/home/gongjing/project/shape_imputation/data/Known_Structures/human_18S.dot' 
    if validate is None: validate = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rfam/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt' 
    if predict is None: predict = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rfam/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.prediction_trainHasNull_lossAll.txt' 

    dots_dict = util.read_dots(dot)
    # print(dots_dict.keys())
    # dot_dict = dots_dict[tx]
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validates = pd.read_csv(validate, header=None, sep='\t')
    df_validates.columns = cols
    df_predict = pd.read_csv(predict, header=None, sep='\t')
    df_validates['fragment_shape(predict)'] = df_predict[0]
    
    tx_ls = list(set(list(df_validates['tx'])))
    roc_auc_dict = nested_dict(2, int)
    for tx in tx_ls:
        # print('process: {}'.format(tx))
    
        df_validate = df_validates[df_validates['tx']==tx]
        # print(df_validate.shape, df_validate.head())
        dot_dict = dots_dict[tx]

        shape_true_dict = nested_dict()
        for v,s,e in zip(df_validate['fragment_shape(true)'], df_validate['start'], df_validate['end']):
            v_ls = v.split(',')
            for n,i in enumerate(range(s,e)):
                shape_true_dict[i] = v_ls[n]
        shape_predict_dict = nested_dict()
        for v,s,e in zip(df_validate['fragment_shape(predict)'], df_validate['start'], df_validate['end']):
            v_ls = v.split(',')
            for n,i in enumerate(range(s,e)):
                shape_predict_dict[i] = v_ls[n]
        shape_true = [float(j) for i,j in sorted(shape_true_dict.items(), key=lambda kv: float(kv[0]))]
        shape_predict = [float(j) for i,j in sorted(shape_predict_dict.items(), key=lambda kv: float(kv[0]))]

        # shape_true = [float(j) for i in df_validate['fragment_shape(true)'] for j in i.split(',')]
        # shape_predict = [float(j) for i in df_validate['fragment_shape(predict)'] for j in i.split(',')]

        shape_true = [np.nan if i == -1 else float(i) for i in shape_true]
        shape_predict = [np.nan if i == -1 else float(i) for i in shape_predict]
        df_true = pd.DataFrame({'reactivity':shape_true})
        df_predict = pd.DataFrame({'reactivity':shape_predict})

        tx_dot = list(dot_dict['dotbracket'].replace('.','1').replace('(','0').replace(')','0').replace('>','0').replace('<','0').replace('[','0').replace(']','0').replace('{','0').replace('}','0'))

        file_list = ['True', 'Predict']
        df_list = [df_true, df_predict]

        # savefn = predict.replace('.txt','.auc.pdf')
        base_used_ls = ':'.join(['ATCG']*len(file_list))
        fpr_list, tpr_list, roc_auc_list, calc_base_num_list = util.calc_auc(file_list, df_list, tx_dot[0+start:df_true.shape[0]+start], None, dot_dict['seq'][0+start:df_true.shape[0]+start], None, None, 0, base_used_ls, 0)
        if roc_auc_list is not None:
            roc_auc_dict[tx]['True'] = roc_auc_list[0]
            roc_auc_dict[tx]['Predict'] = roc_auc_list[1]
            roc_auc_dict[tx]['True(base)'] = calc_base_num_list[0]
            roc_auc_dict[tx]['Predict(base)'] = calc_base_num_list[1]
        
    roc_auc_df = pd.DataFrame.from_dict(roc_auc_dict, orient='index')
    # print(roc_auc_df)
    
    roc_auc_df.to_csv(savefn.replace('.pdf', '.txt'), header=True, index=True, sep='\t')
    
    fig,ax=plt.subplots(figsize=(8,8))
    sns.scatterplot(x='True', y='Predict', data=roc_auc_df, ax=ax)
    ax.set_xlim(0.35,1.05)
    ax.set_ylim(0.35,1.05)
    plt.xticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
    plt.title(title)
    ax.plot((0.2,1.05), (0.2,1.05), 'k-', alpha=0.75, zorder=0)
    plt.tight_layout()
    plt.savefig(savefn)


# known_structures_compare(dot='/home/gongjing/project/shape_imputation/data/Known_Structures/human.dot',tx='ABBA01035051.1/9337-9470', start=0)

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot AUC for batch known structures')
    
    parser.add_argument('--dot', type=str, help='Dot file for known structure')
    parser.add_argument('--validate', type=str, help='Validate fragment file')
    parser.add_argument('--predict', type=str, help='Predicted fragment file')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot')
    parser.add_argument('--title', type=str, default='', help='Title of the plot')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    known_structures_compare(dot=args.dot, validate=args.validate, predict=args.predict, savefn=args.savefn, title=args.title)
    

if __name__ == '__main__':
    main()