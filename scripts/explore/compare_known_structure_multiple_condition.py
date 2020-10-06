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

def known_structure_compare(dot=None, validate=None, predict_ls=None, predict_ls_label=None, tx=None, start=0, savefn=None, title='', bases_ls='ATCG:ATCG:ATCG', save_shapes=1):
    predict_ls = predict_ls.split(':')
    predict_ls_label = predict_ls_label.split(':')
    
    dot_dict = util.read_dot(dot)
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validate = pd.read_csv(validate, header=None, sep='\t')
    df_validate.columns = cols
    df_validate = df_validate[df_validate['tx']==tx]
    
    null_pos = []
    for v1,v2,s,e in zip(df_validate['fragment_shape'], df_validate['fragment_shape(true)'], df_validate['start'], df_validate['end']):
        v_ls1 = v1.split(',')
        v_ls2 = v2.split(',')
        for n,i in enumerate(range(s,e)):
            if v_ls1[n] == '-1' and v_ls2[n] != '-1':
                null_pos.append(i)
    null_pos = list(set(null_pos))
    print(null_pos)
    
    for predict,predict_label in zip(predict_ls, predict_ls_label):
        df_predict = pd.read_csv(predict, header=None, sep='\t')
        df_validate['fragment_shape(predict_{})'.format(predict_label)] = df_predict[0]
    df_validate = df_validate[df_validate['tx']==tx]
    df_validate['fragment_shape(predict_true)'] = df_validate['fragment_shape(true)']
    print(df_validate.shape, df_validate.head())
    
    predict_ls_label = ['true'] + predict_ls_label

    shape_predict_ls = []
    for predict_label in predict_ls_label:
        shape_predict_dict = nested_dict(1,list)
        for v,s,e in zip(df_validate['fragment_shape(predict_{})'.format(predict_label)], df_validate['start'], df_validate['end']):
            v_ls = v.split(',')
            for n,i in enumerate(range(s,e)):
                shape_predict_dict[i].append(float(v_ls[n]))
        shape_predict = [np.mean(jj) for ii,jj in sorted(shape_predict_dict.items(), key=lambda kv: float(kv[0]))]
        shape_predict_ls.append(shape_predict)
    
    # shape_predict_out = savefn.replace('.pdf', '.out')
    # with open(shape_predict_out, 'w') as OUT:
        # OUT.write('\t'.join(map(str, [tx, len(shape_predict), '*']+shape_predict))+'\n')
    
    df_true_predict_ls = []
    for shape_ls in shape_predict_ls:
        shape = [np.nan if i == -1 else float(i) for i in shape_ls]
        df_shape = pd.DataFrame({'reactivity':shape})
        df_true_predict_ls.append(df_shape)
        # print(df_shape.shape, len(null_pos), null_pos)
        # df_true_predict_ls.append(df_shape.iloc[null_pos,:])
    
    tx_dot = list(dot_dict['dotbracket'].replace('.','1').replace('(','0').replace(')','0').replace('>','0').replace('<','0').replace('[','0').replace(']','0').replace('{','0').replace('}','0'))
    
    fpr_list, tpr_list, roc_auc_list, calc_base_num_list = util.calc_auc(predict_ls_label, df_true_predict_ls, tx_dot[0+start:df_true_predict_ls[0].shape[0]+start], None, dot_dict['seq'][0+start:df_true_predict_ls[0].shape[0]+start], savefn, None, 0, bases_ls, 0, title)
    # fpr_list, tpr_list, roc_auc_list, calc_base_num_list = util.calc_auc(file_list, df_list, tx_dot[0+start:df_true.shape[0]+start], a_ls, dot_dict['seq'][0+start:df_true.shape[0]+start], savefn, 3, 0, 'AC', 0, title)
    
#     shape_random_dict = nested_dict(1, list)
#     for v,s,e in zip(df_validate['fragment_shape'], df_validate['start'], df_validate['end']):
#         v_ls = v.split(',')
#         for n,i in enumerate(range(s,e)):
#             shape_random_dict[i].append(float(v_ls[n]))
#     shape_random = [np.mean(j) for i,j in sorted(shape_random_dict.items(), key=lambda kv: float(kv[0]))]
#     shape_random = [np.nan if i == -1 else float(i) for i in shape_random]
    
#     fig,ax=plt.subplots(2,1,figsize=(30,6), sharex=True, sharey=True)
#     ax[0].plot(shape_true, color='red', lw=0.5)
#     ax[0].set_ylabel('true')
#     plt.text(0.98, 0.5, 'AUC=\n{0:.3f}'.format(roc_auc_list[0]), ha='center',va='center', transform=ax[0].transAxes)
#     ax[1].plot(shape_predict, color='blue', lw=0.5)
#     ax[1].set_ylabel('predict')
#     plt.text(0.98, 0.5, 'AUC=\n{0:.3f}'.format(roc_auc_list[1]), ha='center',va='center', transform=ax[1].transAxes)
#     for m,i in enumerate(shape_random):
#             if np.isnan(i):
#                 x1 = m-0.1
#                 x2 = m+0.1
#                 ax[1].axvspan(x1, x2, color='lightgray', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(savefn.replace('.pdf', '.track.pdf'))
#     plt.close()
    
    if save_shapes:
        # df_shapes_ls = []
        # for i,j in zip(predict_ls_label, df_true_predict_ls):
    #         j.columns = [i]
    #         # df_shapes_ls.append(j)
        df_shapes = pd.concat(df_true_predict_ls, axis=1)
        df_shapes.to_csv(savefn.replace('.pdf', '.shape_reactivity.txt'), header=True, index=False, sep='\t')

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot AUC for single known structure')
    
    parser.add_argument('--dot', type=str, default='/home/gongjing/project/shape_imputation/data/Known_Structures/human_18S.dot', help='Dot file for known structure')
    parser.add_argument('--validate', type=str, default='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation_randomNULL0.1.txt', help='Validate fragment file')
    parser.add_argument('--predict_ls', type=str, default='/home/gongjing/project/shape_imputation/exper/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate2/prediction.rRNA.txt:/home/gongjing/project/shape_imputation/exper/b91_trainLossall_seqOnly_x10/prediction.rRNA.txt:/home/gongjing/project/shape_imputation/exper/b92_trainLossall_shapeOnly_x10/prediction.rRNA.txt', help='Predicted fragment file')
    parser.add_argument('--predict_ls_label', type=str, default='seq+shape:seq:shape', help='Label of predicted fragment file')
    parser.add_argument('--tx', type=str, default='18S', help='Transcript to plot')
    parser.add_argument('--start', type=int, default=0, metavar='N',help='Dot start index')
    parser.add_argument('--savefn', type=str, default='/home/gongjing/project/shape_imputation/results/condition_compare_AUC_18S.pdf', help='Pdf file to save plot')
    parser.add_argument('--title', type=str, default='', help='Title of the plot')
    parser.add_argument('--bases_ls', type=str, default='ATCGU:ATCGU:ATCGU:ATCGU', help='Bases considered while calc AUC for predict sample')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    known_structure_compare(dot=args.dot, validate=args.validate, predict_ls=args.predict_ls, predict_ls_label=args.predict_ls_label, tx=args.tx, start=args.start, savefn=args.savefn, title=args.title, bases_ls=args.bases_ls)
    
if __name__ == '__main__':
    main()