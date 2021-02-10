import pandas as pd
import numpy as np
import util
import sys,os

import argparse
from collections import OrderedDict
import subprocess

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"
from nested_dict import nested_dict

def generate_evaluate_shell(d):
    evaluate_savefn = d + '/evaluate.sh'
    SAVEFN = open(evaluate_savefn, 'w')
    train_shell = d + '/train.sh'
    
    conf_dict = config()
    conf_dict['exper_dir'] = d
    
    model_parameter_dict = OrderedDict()
    with open(train_shell, 'r') as TRAIN:
        for line in TRAIN:
            line = line.strip('\n')
            line = line.replace('RNA-structure-profile-imputation', 'ShapeImputation')
            if 'CUDA_VISIBLE_DEVICES' in line:
                continue
            elif '--' not in line:
                SAVEFN.write(line+'\n')
            else:
                arr = line.strip(' ').split(' ')
                arr[1] = '' if len(arr) == 2 else arr[1]
                model_parameter_dict[arr[0]] = arr[1]
                print(arr)
        model_parameter_dict['--loaded_pt_file'] = '$work_space/prediction.pt'
        filename_validation = model_parameter_dict['--filename_validation']
        model_parameter_dict['--filename_validation'] = '$2'
        model_parameter_dict['--filename_prediction'] = '$work_space/prediction.$3.txt'
        model_parameter_dict['--logfile'] = '$work_space/log.$3.txt'
        
        predict_fun_str = write_predict_fun(model_parameter_dict, conf_dict)
        SAVEFN.write(predict_fun_str+'\n')
        
        predict_str = write_predict_sample(conf_dict)
        SAVEFN.write(predict_str+'\n')
        
        CLIP_str = write_CLIP(conf_dict, extend=10, motif_len=7)
        SAVEFN.write(CLIP_str+'\n')
        
        m6A_str = write_m6A(conf_dict, extend=10, motif_len=1)
        SAVEFN.write(m6A_str+'\n')
        
        start_stop_codon_str = write_start_stop_codon(conf_dict, extend=50, motif_len=0)
        SAVEFN.write(start_stop_codon_str+'\n')
        
        rRNA_str = write_rRNA(conf_dict)
        SAVEFN.write(rRNA_str+'\n')
        
        rfam_str = write_rfam(conf_dict)
        SAVEFN.write(rfam_str+'\n')
        
        rep_str = write_rep(conf_dict)
        SAVEFN.write(rep_str+'\n')
        
        explore_str = write_compare_true_predict_corr(filename_validation, conf_dict)
        SAVEFN.write(explore_str+'\n')
        
        merge_str = write_merge_pdf()
        SAVEFN.write(merge_str+'\n')
        
        DMSseq_str = write_DMSseq(conf_dict)
        SAVEFN.write(DMSseq_str+'\n')
        
        rRNA_18S_pct_dtr = write_18S_different_null_pct(conf_dict)
        SAVEFN.write(rRNA_18S_pct_dtr+'\n')
    SAVEFN.close()
    
    return evaluate_savefn
  
def write_predict_fun(model_parameter_dict, conf_dict):
    fun_str = ''
    fun_str += '''predict(){\necho "predict: "$2", "$3\ntime CUDA_VISIBLE_DEVICES=$1 python $script_dir/main.py --load_model_and_predict \\\n'''
    for k,v in model_parameter_dict.items():
        if v == ' ': v = ''
        if k == '--logfile':
            fun_str += ' '*8+k+' '+v+'\n'
        else:
            fun_str += ' '*8+k+' '+v+' \\\n'
    fun_str += '}\n'
    
    fun_str += '\n'
    fun_str += 'compare_known_structure={}'.format(conf_dict['compare_known_structure']) + '\n'
    fun_str += 'compare_known_structures={}'.format(conf_dict['compare_known_structures']) + '\n'
    fun_str += 'rep_compare={}'.format(conf_dict['rep_compare']) + '\n'
    fun_str += 'compare_true_and_predict={}'.format(conf_dict['compare_true_and_predict']) + '\n'
    fun_str += 'plot_predict_heatmap={}'.format(conf_dict['plot_predict_heatmap']) + '\n'
    return fun_str

def write_predict_sample(conf_dict):
    predict_str = ''
    predict_str += '{}={}'.format('validation_rRNA', conf_dict['validation_rRNA']) + '\n'
    predict_str += '{}={}'.format('validation_rfam', conf_dict['validation_rfam']) + '\n'
    predict_str += '{}={}'.format('validation_rep1', conf_dict['validation_rep1']) + '\n'
    predict_str += '{}={}'.format('validation_rep2', conf_dict['validation_rep2']) + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_rRNA', 'rRNA') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_rfam', 'rfam') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_rep1', 'rep1') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_rep2', 'rep2') + '\n'
    
    predict_str += '{}={}'.format('validation_hek_ch_vivo', conf_dict['validation_hek_ch_vivo0.1']) + '\n'
    predict_str += '{}={}'.format('validation_hek_cy_vivo', conf_dict['validation_hek_cy_vivo0.1']) + '\n'
    predict_str += '{}={}'.format('validation_hek_np_vivo', conf_dict['validation_hek_np_vivo0.1']) + '\n'
    predict_str += '{}={}'.format('validation_mes_wc_vivo', conf_dict['validation_mes_wc_vivo0.1']) + '\n'
    predict_str += '{}={}'.format('validation_hek_wc_vitro', conf_dict['validation_hek_wc_vitro0.1']) + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_hek_ch_vivo', 'hek_ch_vivo0.1') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_hek_cy_vivo', 'hek_cy_vivo0.1') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_hek_np_vivo', 'hek_np_vivo0.1') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_mes_wc_vivo', 'mes_wc_vivo0.1') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_hek_wc_vitro', 'hek_wc_vitro0.1') + '\n'
    
    return predict_str

def write_CLIP(conf_dict, extend=10, motif_len=7):
    predict_str = ''
    predict_str += '{}={}'.format('validation_FXR_exceed', conf_dict['validation_FXR_exceed']) + '\n'
    predict_str += '{}={}'.format('validation_FXR_ok', conf_dict['validation_FXR_ok']) + '\n'
    predict_str += '{}={}'.format('validation_LIN28_exceed', conf_dict['validation_LIN28_exceed']) + '\n'
    predict_str += '{}={}'.format('validation_LIN28_ok', conf_dict['validation_LIN28_ok']) + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_FXR_exceed', 'FXR_exceed') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_FXR_ok', 'FXR_ok') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_LIN28_exceed', 'LIN28_exceed') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_LIN28_ok', 'LIN28_ok') + '\n'
    predict_str += '''python $plot_predict_heatmap --predict_ls $work_space/prediction.FXR_exceed.txt:$work_space/prediction.FXR_ok.txt \\\n {}--label_ls FXR_exceed:FXR_ok \\\n {}--extend {} \\\n {}--motif_len {} \\\n {}--savefn_meta $work_space/prediction.FXR.pdf'''.format(' '*8, ' '*8, extend, ' '*8, motif_len, ' '*8,) + '\n'
    predict_str += '''python $plot_predict_heatmap --predict_ls $work_space/prediction.LIN28_exceed.txt:$work_space/prediction.LIN28_ok.txt \\\n {}--label_ls LIN28_exceed:LIN28_ok \\\n {}--extend {} \\\n {}--motif_len {} \\\n {}--savefn_meta $work_space/prediction.LIN28.pdf'''.format(' '*8, ' '*8, extend, ' '*8, motif_len, ' '*8,) + '\n'
    return predict_str

def write_m6A(conf_dict, extend=10, motif_len=1):
    predict_str = ''
    predict_str += '{}={}'.format('validation_m6A_null', conf_dict['validation_m6A_null']) + '\n'
    predict_str += '{}={}'.format('validation_m6A_valid', conf_dict['validation_m6A_valid']) + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_m6A_null', 'm6A_null') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_m6A_valid', 'm6A_valid') + '\n'
    predict_str += '''python $plot_predict_heatmap --predict_ls $work_space/prediction.m6A_null.txt:$work_space/prediction.m6A_valid.txt \\\n {}--label_ls m6A_null:m6A_valid \\\n {}--extend {} \\\n {}--motif_len {} \\\n {}--savefn_meta $work_space/prediction.m6A.pdf'''.format(' '*8, ' '*8, extend, ' '*8, motif_len, ' '*8,) + '\n'
    return predict_str
    
def write_start_stop_codon(conf_dict, extend=50, motif_len=0):
    predict_str = ''
    predict_str += '{}={}'.format('validation_start_codon_null', conf_dict['validation_start_codon_null']) + '\n'
    predict_str += '{}={}'.format('validation_start_codon_valid', conf_dict['validation_start_codon_valid']) + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_start_codon_null', 'start_codon_null') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_start_codon_valid', 'start_codon_valid') + '\n'
    predict_str += '''python $plot_predict_heatmap --predict_ls $work_space/prediction.start_codon_null.txt:$work_space/prediction.start_codon_valid.txt \\\n {}--label_ls start_codon_null:start_codon_valid \\\n {}--extend {} \\\n {}--motif_len {} \\\n {}--savefn_meta $work_space/prediction.start_codon.pdf'''.format(' '*8, ' '*8, extend, ' '*8, motif_len, ' '*8,) + '\n'
    
    predict_str += '{}={}'.format('validation_stop_codon_null', conf_dict['validation_stop_codon_null']) + '\n'
    predict_str += '{}={}'.format('validation_stop_codon_valid', conf_dict['validation_stop_codon_valid']) + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_stop_codon_null', 'stop_codon_null') + '\n'
    predict_str += '''predict {} {} "{}"'''.format('$1', '$validation_stop_codon_valid', 'stop_codon_valid') + '\n'
    predict_str += '''python $plot_predict_heatmap --predict_ls $work_space/prediction.stop_codon_null.txt:$work_space/prediction.stop_codon_valid.txt \\\n {}--label_ls stop_codon_null:stop_codon_valid \\\n {}--extend {} \\\n {}--motif_len {} \\\n {}--savefn_meta $work_space/prediction.stop_codon.pdf'''.format(' '*8, ' '*8, extend, ' '*8, motif_len, ' '*8,) + '\n'
    
    return predict_str
    
def write_rRNA(conf_dict):
    rRNA_str = ''
    # rRNA_str += 'compare_known_structure={}'.format(conf_dict['compare_known_structure']) + '\n'
    rRNA_str += 'human_18S_dot={}'.format(conf_dict['human_18S_dot']) + '\n'
    rRNA_str += 'human_28S_dot={}'.format(conf_dict['human_28S_dot']) + '\n'
    rRNA_str += '''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_rRNA \\\n {}--predict $work_space/prediction.rRNA.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.rRNA.18S_AUC.pdf \\\n {}--title 18S'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    rRNA_str += '''python $compare_known_structure --dot $human_28S_dot \\\n {}--validate $validation_rRNA \\\n {}--predict $work_space/prediction.rRNA.txt \\\n {}--tx 28S \\\n {}--start 157 \\\n {}--savefn $work_space/prediction.rRNA.28S_AUC.pdf \\\n {}--title 28S'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    
    return rRNA_str

def write_18S_different_null_pct(conf_dict):
    rRNA_18S_pct_dtr = ''
    rRNA_18S_pct_dtr += 'for p in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5' + '\n'
    rRNA_18S_pct_dtr += 'do' + '\n'
    rRNA_18S_pct_dtr += ' '*4 + 'validation_18S=/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation_randomNULL$p.txt' + '\n'
    rRNA_18S_pct_dtr += ' '*4 + 'predict $1 $validation_18S 18S$p' + '\n'
    rRNA_18S_pct_dtr += ' '*4 +'''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_18S \\\n {}--predict $work_space/prediction.18S$p.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.18S$p.AUC.pdf \\\n {}--title "18S(null="$p")"'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    rRNA_18S_pct_dtr += 'done' +'\n'
    rRNA_18S_pct_dtr += 'montage prediction.18S{0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5}.AUC.pdf -mode concatenate merge.18S.pdf' + '\n'
    
    rRNA_18S_pct_dtr += 'rRNA_18S_track_plot={}'.format(conf_dict['rRNA_18S_track_plot']) + '\n'
    rRNA_18S_pct_dtr += '''python $rRNA_18S_track_plot --exper_dir {}'''.format(conf_dict['exper_dir']) + '\n'
    
    
    return rRNA_18S_pct_dtr

def write_DMSseq(conf_dict):
    DMSseq_str = ''
    DMSseq_str += '{}={}'.format('validation_DMSseq_fibroblast_vivo', conf_dict['validation_DMSseq_fibroblast_vivo']) + '\n'
    DMSseq_str += '{}={}'.format('validation_DMSseq_fibroblast_vivo_rRNA', conf_dict['validation_DMSseq_fibroblast_vivo_rRNA']) + '\n'
    DMSseq_str += '{}={}'.format('validation_DMSseq_K562_vivo_rRNA', conf_dict['validation_DMSseq_K562_vivo_rRNA']) + '\n'
    DMSseq_str += '''predict {} {} "{}"'''.format('$1', '$validation_DMSseq_fibroblast_vivo', 'DMSseq_fibroblast_vivo') + '\n'
    DMSseq_str += '''predict {} {} "{}"'''.format('$1', '$validation_DMSseq_fibroblast_vivo_rRNA', 'DMSseq_fibroblast_vivo_rRNA') + '\n'
    DMSseq_str += '''predict {} {} "{}"'''.format('$1', '$validation_DMSseq_K562_vivo_rRNA', 'DMSseq_K562_vivo_rRNA') + '\n'
    DMSseq_str += '''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_DMSseq_fibroblast_vivo_rRNA \\\n {}--predict $work_space/prediction.DMSseq_fibroblast_vivo_rRNA.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.DMSseq_fibroblast_vivo_rRNA.18S_AUC.pdf \\\n {}--title DMSseq_fibroblast_vivo_rRNA_18S \\\n {}--predict_bases ATCG \\\n {}--validate_bases ATCG'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    DMSseq_str += '''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_DMSseq_K562_vivo_rRNA \\\n {}--predict $work_space/prediction.DMSseq_K562_vivo_rRNA.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.DMSseq_K562_vivo_rRNA.18S_AUC.pdf \\\n {}--title DMSseq_K562_vivo_rRNA_18S \\\n {}--predict_bases ATCG \\\n {}--validate_bases ATCG'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    DMSseq_str += '''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_DMSseq_fibroblast_vivo_rRNA \\\n {}--predict $work_space/prediction.DMSseq_fibroblast_vivo_rRNA.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.DMSseq_fibroblast_vivo_rRNA.18S_AUC.AC.pdf \\\n {}--title DMSseq_fibroblast_vivo_rRNA_18S \\\n {}--predict_bases AC \\\n {}--validate_bases AC'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    DMSseq_str += '''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_DMSseq_K562_vivo_rRNA \\\n {}--predict $work_space/prediction.DMSseq_K562_vivo_rRNA.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.DMSseq_K562_vivo_rRNA.18S_AUC.AC.pdf \\\n {}--title DMSseq_K562_vivo_rRNA_18S \\\n {}--predict_bases AC \\\n {}--validate_bases AC'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    DMSseq_str += '''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_DMSseq_fibroblast_vivo_rRNA \\\n {}--predict $work_space/prediction.DMSseq_fibroblast_vivo_rRNA.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.DMSseq_fibroblast_vivo_rRNA.18S_AUC.TG.pdf \\\n {}--title DMSseq_fibroblast_vivo_rRNA_18S \\\n {}--predict_bases TG \\\n {}--validate_bases AC'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    DMSseq_str += '''python $compare_known_structure --dot $human_18S_dot \\\n {}--validate $validation_DMSseq_K562_vivo_rRNA \\\n {}--predict $work_space/prediction.DMSseq_K562_vivo_rRNA.txt \\\n {}--tx 18S \\\n {}--start 0 \\\n {}--savefn $work_space/prediction.DMSseq_K562_vivo_rRNA.18S_AUC.TG.pdf \\\n {}--title DMSseq_K562_vivo_rRNA_18S \\\n {}--predict_bases TG \\\n {}--validate_bases AC'''.format(' '*8, ' '*8,' '*8,' '*8,' '*8,' '*8,' '*8,' '*8) + '\n'
    DMSseq_str += '''python $compare_true_and_predict --validate $validation_DMSseq_fibroblast_vivo --predict {} --plot_savefn {}'''.format('./prediction.DMSseq_fibroblast_vivo.txt', './prediction.DMSseq_fibroblast_vivo.explore.pdf') + '\n'
    DMSseq_str += '''python $compare_true_and_predict --validate $validation_DMSseq_fibroblast_vivo_rRNA --predict {} --plot_savefn {}'''.format('./prediction.DMSseq_fibroblast_vivo_rRNA.txt', './prediction.DMSseq_fibroblast_vivo_rRNA.explore.pdf') + '\n'
    
    return DMSseq_str

def write_rfam(conf_dict):
    rfam_str = ''
    # rfam_str += 'compare_known_structures={}'.format(conf_dict['compare_known_structures']) + '\n'
    rfam_str += 'human_rfam_dot={}'.format(conf_dict['human_rfam_dot']) + '\n'
    rfam_str += '''python $compare_known_structures --dot $human_rfam_dot \\\n {}--validate $validation_rfam \\\n {}--predict $work_space/prediction.rfam.txt \\\n {}--savefn $work_space/prediction.rfam.AUCs.pdf \\\n {}--title rfam'''.format(' '*8, ' '*8,' '*8, ' '*8) + '\n'

    return rfam_str

def write_rep(conf_dict):
    rep_str = ''
    # rep_str += 'rep_compare={}'.format(conf_dict['rep_compare']) + '\n'
    rep_str += 'rep1_out={}'.format(conf_dict['rep1_out']) + '\n'
    rep_str += 'rep2_out={}'.format(conf_dict['rep2_out']) + '\n'
    rep_str += '''python $rep_compare --rep1_out $rep1_out \\\n {}--rep1_validate $validation_rep1 \\\n {}--rep1_predict $work_space/prediction.rep1.txt \\\n {}--rep2_out $rep2_out \\\n {}--rep2_validate $validation_rep2 \\\n {}--rep2_predict $work_space/prediction.rep2.txt \\\n {}--tx_null_pct 0.3 \\\n {}--savefn $work_space/prediction.rep_corr.pdf'''.format(' '*8, ' '*8,' '*8,' '*8, ' '*8,' '*8,' '*8, ) + '\n'
    
    return rep_str

def write_compare_true_predict_corr(filename_validation, conf_dict):
    explore_str = ''
    # explore_str += 'compare_true_and_predict={}'.format(conf_dict['compare_true_and_predict']) + '\n'
    explore_str += 'filename_validation={}'.format(filename_validation) + '\n'
    explore_str += '''python $compare_true_and_predict --validate $filename_validation --predict {} --plot_savefn {}'''.format('./prediction.txt', './prediction.explore.pdf') + '\n'
    
    return explore_str
    
def write_merge_pdf():
    """ montage prediction.explore.frag_corr.pdf prediction.rRNA.18S_AUC.pdf prediction.rRNA.28S_AUC.pdf prediction.rfam.AUCs.pdf prediction.rep_corr.pdf -mode concatenate merge.pdf
    """
    merge_str = ''
    merge_str += 'montage '
    merge_str += 'prediction.explore.frag_corr.pdf '
    merge_str += 'prediction.rRNA.18S_AUC.pdf '
    merge_str += 'prediction.rRNA.28S_AUC.pdf '
    merge_str += 'prediction.rfam.AUCs.pdf '
    merge_str += 'prediction.rep_corr.pdf '
    merge_str += '-mode concatenate merge.pdf' +'\n'
    
    return merge_str
    
def config():
    conf_dict = {}
    conf_dict['validation_rRNA'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation_randomNULL0.1.txt'
    conf_dict['validation_rfam'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rfam/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt'
    conf_dict['validation_rep1'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep1/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt'
    conf_dict['validation_rep2'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep2/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt'
    conf_dict['validation_DMSseq_fibroblast_vivo'] = '/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation.txt'
    conf_dict['validation_DMSseq_fibroblast_vivo_rRNA'] = '/home/gongjing/project/shape_imputation/data/DMSseq_fibroblast_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation.txt'
    conf_dict['validation_DMSseq_K562_vivo_rRNA'] = '/home/gongjing/project/shape_imputation/data/DMSseq_K562_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation.txt'
    
    conf_dict['compare_known_structure'] = '/home/gongjing/project/shape_imputation/ShapeImputation/scripts/explore/compare_known_structure_resplit.py'
    conf_dict['human_18S_dot'] = '/home/gongjing/project/shape_imputation/data/Known_Structures/human_18S.dot'
    conf_dict['human_28S_dot'] = '/home/gongjing/project/shape_imputation/data/Known_Structures/human_28S.dot'
    conf_dict['validation_rRNA'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength.validation_randomNULL0.1.txt'
    conf_dict['validation_FXR_exceed'] = '/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt.sort.shape100.txt'
    conf_dict['validation_FXR_ok'] = '/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.FXR.txt.sort.shape100.txt'
    conf_dict['validation_LIN28_exceed'] = '/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20173_LIN28A_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.LIN28.txt.sort.shape100.txt'
    conf_dict['validation_LIN28_ok'] = '/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20173_LIN28A_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.LIN28.txt.sort.shape100.txt'
    conf_dict['validation_m6A_valid'] = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_valid.e0.bed.sort.shape100.txt'
    conf_dict['validation_m6A_null'] = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_null.e0.bed.sort.shape100.txt'
    conf_dict['validation_start_codon_valid'] = '/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.start_codon.ok.sort.shape'
    conf_dict['validation_start_codon_null'] = '/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.start_codon.null.sort.shape'
    conf_dict['validation_stop_codon_valid'] = '/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.stop_codon.ok.sort.shape'
    conf_dict['validation_stop_codon_null'] = '/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.stop_codon.null.sort.shape'
    
    conf_dict['validation_hek_ch_vivo0.1'] = '/home/gongjing/project/shape_imputation/data/hek_ch_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt'
    conf_dict['validation_hek_cy_vivo0.1'] = '/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt'
    conf_dict['validation_hek_np_vivo0.1'] = '/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt'
    conf_dict['validation_mes_wc_vivo0.1'] = '/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt'
    conf_dict['validation_hek_wc_vitro0.1'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_randomNULL0.1.txt'
    
    conf_dict['compare_known_structures'] = '/home/gongjing/project/shape_imputation/ShapeImputation/scripts/explore/compare_known_structures.py'
    conf_dict['human_rfam_dot'] = '/home/gongjing/project/shape_imputation/data/Known_Structures/human.dot'
    
    conf_dict['rep_compare'] = '/home/gongjing/project/shape_imputation/ShapeImputation/scripts/explore/compare_rep_corr.py'
    conf_dict['rep1_out'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep1/shape.c200T2M0m0.out'
    conf_dict['rep2_out'] = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape_rep2/shape.c200T2M0m0.out'
    
    conf_dict['compare_true_and_predict'] = '/home/gongjing/project/shape_imputation/ShapeImputation/scripts/explore/compare_true_and_predict.py'
    conf_dict['rRNA_18S_track_plot'] = '/home/gongjing/project/shape_imputation/ShapeImputation/scripts/explore/rRNA_18S_track_plot.py'
    
    conf_dict['plot_predict_heatmap'] = '/home/gongjing/project/shape_imputation/ShapeImputation/scripts/explore/plot_predict_heatmap.py'
    
    return conf_dict

def run_evaluate_shell(evaluate_savefn, GPU_id=0):
    evaluate_dir = os.path.dirname(evaluate_savefn)
    subprocess.call(["cd {}; bash {} {}".format(evaluate_dir, evaluate_savefn, GPU_id)], shell=True)

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Generate evalaute.sh and run')
    
    parser.add_argument('--d', type=str, help='Model dir')
    parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID used')
    
    # get args
    args = parser.parse_args()
    evaluate_savefn = generate_evaluate_shell(d=args.d)
    run_evaluate_shell(evaluate_savefn, GPU_id=args.GPU_id)

if __name__ == '__main__':
    main()