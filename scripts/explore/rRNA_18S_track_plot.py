import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
from nested_dict import nested_dict
import argparse
import util
from scipy import stats

def known_structure_compare(dot=None, validate=None, predict=None, tx=None, start=0, savefn=None):
    if dot is None: dot = '/home/gongjing/project/shape_imputation/data/Known_Structures/human_18S.dot' 
    if validate is None: validate = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation_randomNULL0.1.txt' 
    if predict is None: predict = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation_randomNULL0.1.prediction_trainHasNull_lossAll.txt' 
    if tx is None: tx = '18S'
    print(dot,validate,predict,tx,start,savefn)
    
    dot_dict = util.read_dot(dot)
    cols = ['tx', 'length', 'start', 'end', 'mean_reactivity', 'null_pct','seq','fragment_shape', 'fragment_shape(true)']
    df_validate = pd.read_csv(validate, header=None, sep='\t')
    df_validate.columns = cols
    df_predict = pd.read_csv(predict, header=None, sep='\t')
    df_validate['fragment_shape(predict)'] = df_predict[0]
    df_validate = df_validate[df_validate['tx']==tx]
#     print(df_validate.shape, df_validate.head())
    
    shape_random_dict = nested_dict(1, list)
    for v,s,e in zip(df_validate['fragment_shape'], df_validate['start'], df_validate['end']):
        v_ls = v.split(',')
        for n,i in enumerate(range(s,e)):
            shape_random_dict[i].append(float(v_ls[n]))
    shape_true_dict = nested_dict(1, list)
    for v,s,e in zip(df_validate['fragment_shape(true)'], df_validate['start'], df_validate['end']):
        v_ls = v.split(',')
        for n,i in enumerate(range(s,e)):
            shape_true_dict[i].append(float(v_ls[n]))
    shape_predict_dict = nested_dict(1, list)
    for v,s,e in zip(df_validate['fragment_shape(predict)'], df_validate['start'], df_validate['end']):
        v_ls = v.split(',')
        for n,i in enumerate(range(s,e)):
            shape_predict_dict[i].append(float(v_ls[n]))
            
    shape_random = [np.mean(j) for i,j in sorted(shape_random_dict.items(), key=lambda kv: float(kv[0]))]
    shape_true = [np.mean(j) for i,j in sorted(shape_true_dict.items(), key=lambda kv: float(kv[0]))]
    shape_predict = [np.mean(j) for i,j in sorted(shape_predict_dict.items(), key=lambda kv: float(kv[0]))]
    
    shape_random = [np.nan if i == -1 else float(i) for i in shape_random]
    shape_true = [np.nan if i == -1 else float(i) for i in shape_true]
    shape_predict = [np.nan if i == -1 else float(i) for i in shape_predict]
    
    
    df_true = pd.DataFrame({'reactivity':shape_true})
    df_predict = pd.DataFrame({'reactivity':shape_predict})
    
    tx_dot = list(dot_dict['dotbracket'].replace('.','1').replace('(','0').replace(')','0').replace('>','0').replace('<','0').replace('[','0').replace(']','0').replace('{','0').replace('}','0'))
    file_list = ['True', 'Predict']
    df_list = [df_true, df_predict]
    base_used_ls = ':'.join(['ATCG']*len(file_list))
    fpr_list, tpr_list, roc_auc_list, calc_base_num_list = util.calc_auc(file_list, df_list, tx_dot[0+start:df_true.shape[0]+start], None, dot_dict['seq'][0+start:df_true.shape[0]+start], None, None, 0, base_used_ls, 0, '')
    print("all", roc_auc_list)
    
    shape_valid_pos_access = [-1 if np.isnan(i) else 1 for i in shape_random]
    validpos_fpr_list, validpos_tpr_list, validpos_roc_auc_list, validpos_calc_base_num_list = util.calc_auc(file_list, df_list, tx_dot[0+start:df_true.shape[0]+start], shape_valid_pos_access, dot_dict['seq'][0+start:df_true.shape[0]+start], None, 0, 0, base_used_ls, 0, '')
    print("valid pos", validpos_roc_auc_list, validpos_calc_base_num_list)
    shape_null_pos_access = [1 if np.isnan(i) else -1 for i in shape_random]
    nullpos_fpr_list, nullpos_tpr_list, nullpos_roc_auc_list, nullpos_calc_base_num_list = util.calc_auc(file_list, df_list, tx_dot[0+start:df_true.shape[0]+start], shape_null_pos_access, dot_dict['seq'][0+start:df_true.shape[0]+start], None, 0, 0, base_used_ls, 0, '')
    print("null pos", nullpos_roc_auc_list, nullpos_calc_base_num_list)
    
    return shape_random,shape_true,shape_predict,roc_auc_list,validpos_roc_auc_list,nullpos_roc_auc_list

def plot_multi_track(validation_18S=None, p_ls=None, exper_dir=None, start=0, end=2000, savefn=None):
    if validation_18S is None:
        validation_18S='/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL0.1.txt'
    if p_ls is None:
        p_ls = [0.1,0.2,0.3,0.4,0.5]

    shape_dict = nested_dict()

    for p in p_ls:
        validation_18S = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL{}.txt'.format(p)
        predict_18S ='{}/prediction.18S{}.txt'.format(exper_dir,p)
        # print("predict_18S", predict_18S)
        shape_random,shape_true,shape_predict,roc_auc_list,validpos_roc_auc_list,nullpos_roc_auc_list = known_structure_compare(dot=None, validate=validation_18S, predict=predict_18S, tx='18S', start=0, savefn=None)
        
        shape_dict[p]['random'] = shape_random
        shape_dict[p]['true'] = shape_true
        shape_dict[p]['predict'] = shape_predict
        shape_dict[p]['AUC'] = roc_auc_list[1] # [true,predict]
        shape_dict[p]['AUC(validpos)'] = validpos_roc_auc_list[1] #
        shape_dict[p]['AUC(nullpos)'] = nullpos_roc_auc_list[1] # 

    n_ax = len(shape_dict) + 1

    fig,ax=plt.subplots(n_ax,1,figsize=(30,6*3), sharex=True, sharey=True)

    ax[0].plot(shape_dict[p_ls[0]]['true'][start:end], color='red', lw=0.5)
    ax[0].set_ylabel('true')
    plt.text(0.98, 0.5, 'AUC=\n{0:.3f}'.format(roc_auc_list[0]), ha='center',va='center', transform=ax[0].transAxes)

    for n,p in enumerate(p_ls):
        ax[n+1].plot(shape_dict[p]['predict'][start:end], color='blue', lw=0.5)
#         ax[n+1].plot(shape_dict[p_ls[0]]['true'][start:end], color='red', lw=0.5)
        diff = [i-j for i,j in zip(shape_dict[p_ls[0]]['true'][start:end],shape_dict[p]['predict'][start:end])]
        # ax[n+1].plot(diff, color='red', lw=0.5)
        
        corr_all_r,  corr_all_p = stats.pearsonr([i for i,j in zip(shape_dict[p_ls[0]]['true'][start:end], shape_dict[p]['predict'][start:end]) if not np.isnan(i) and not np.isnan(j)], [j for i,j in zip(shape_dict[p_ls[0]]['true'][start:end], shape_dict[p]['predict'][start:end]) if not np.isnan(i) and not np.isnan(j)])
        # print([j for i,j in zip(shape_dict[p]['random'][start:end], shape_dict[p_ls[0]]['true'][start:end]) if np.isnan(i)])
        corr_null_r,  corr_null_p = stats.pearsonr([j for i,j,k in zip(shape_dict[p]['random'][start:end], shape_dict[p_ls[0]]['true'][start:end], shape_dict[p]['predict'][start:end]) if np.isnan(i) and not np.isnan(j) and not np.isnan(k)], [k for i,j,k in zip(shape_dict[p]['random'][start:end], shape_dict[p_ls[0]]['true'][start:end], shape_dict[p]['predict'][start:end]) if np.isnan(i) and not np.isnan(j) and not np.isnan(k)])
        # print("calc pearsonr", corr_all_r,  corr_all_p)

        null_num = len([i for i in shape_dict[p]['random'][start:end] if np.isnan(i)])
        ax[n+1].set_ylabel('p({})\nnull({})'.format(p, null_num))
        # x = 'AUC=\n{:.3f}\nr={:.3f}'.format(shape_dict[p]['AUC'], corr_all_r)
        print(shape_dict[p]['AUC'], corr_all_r, corr_null_r, shape_dict[p]['AUC(validpos)'], shape_dict[p]['AUC(nullpos)'])
        plt.text(0.98, 0.5, 'AUC=\n{:.3f}\nr_all={:.3f}\nt_null={:.3f}'.format(shape_dict[p]['AUC'], corr_all_r, corr_null_r), ha='center',va='center', transform=ax[n+1].transAxes)

        for m,i in enumerate(shape_dict[p]['random'][start:end]):
            if np.isnan(i):
                x1 = m-0.1
                x2 = m+0.1
                ax[n+1].axvspan(x1, x2, color='lightgray', alpha=0.3)
            # if m%100 == 0: ax[n+1].axvline(x=m, ymin=0, ymax=1, color='black')
               
    plt.tight_layout()
    plt.savefig(exper_dir+'/prediction.18S.tracks.pdf')
    plt.close()
                
                
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Plot multiple tracks along 18S')
    
    parser.add_argument('--exper_dir', type=str, default='/home/gongjing/project/shape_imputation/exper/31_trainLossall_GmultiplyX', help='Dir of work experiments')
    parser.add_argument('--p_ls', type=str, default='0.1,0.2,0.3,0.4,0.5', help='percentage to plot')
    parser.add_argument('--start', type=int, default=0, help='Plot start index')
    parser.add_argument('--end', type=int, default=2000, help='Plot end index')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot')

    
    # get args
    args = parser.parse_args()
    
    plot_multi_track(exper_dir=args.exper_dir, p_ls=args.p_ls.split(','), start=args.start, end=args.end, savefn=args.savefn)
    

if __name__ == '__main__':
    main()