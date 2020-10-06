import pandas as pd
import numpy as np
import util
import subprocess
import generate_data_set
    
# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.txt'
# rpkm = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.exp_vs_null.txt'

# # fragment_data_split_by_rpkm(fragment, rpkm)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train_randomNULL.txt'
# rpkm = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.exp_vs_null.txt'

# fragment_data_split_by_rpkm(fragment, rpkm)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validationNoRandom.txt'
# np.random.seed(1234)
# util.data_random_null(fragment, null_pct=0.1)
# util.data_random_null(fragment, null_pct=0.2)
# util.data_random_null(fragment, null_pct=0.3)
# util.data_random_null(fragment, null_pct=0.4)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_rRNA/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.fulllength18S.validation.txt'
# np.random.seed(1234)
# util.fragment_split(fragment=fragment, train_frac=0.8)
# util.data_random_null(fragment, null_pct=0.1, col=9)
# util.data_random_null(fragment, null_pct=0.15, col=9)
# util.data_random_null(fragment, null_pct=0.2, col=9)
# util.data_random_null(fragment, null_pct=0.25, col=9)
# util.data_random_null(fragment, null_pct=0.3, col=9)
# util.data_random_null(fragment, null_pct=0.35, col=9)
# util.data_random_null(fragment, null_pct=0.4, col=9)
# util.data_random_null(fragment, null_pct=0.45, col=9)
# util.data_random_null(fragment, null_pct=0.5, col=9)

# fragment = '/data/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_x10_then_pct30_maxL20/windowLen100.sliding100.trainx10.txt'
# np.random.seed(1234)
# util.data_random_null(fragment, null_pct=0.3, col=9)

# for i in ['hek_ch_vivo', 'hek_cy_vivo', 'hek_np_vivo', 'hek_wc_vitro']:
    # fragment = '/home/gongjing/project/shape_imputation/data/{}/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.inwc6205.txt'.format(i)
# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull.inwc6205.txt'
# fragment = '/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.train+validation_truenull.txt'
# fragment = '/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation_truenull_rmnull.txt'
# np.random.seed(1234)
# savefn = fragment.replace('truenull', 'truenull_randomNULL0.3')
# util.data_random_null(fragment, null_pct=0.3, col=9, savefn=savefn)

# seed_ls = [1234, 9999, 5678, 12315, 400100, 42, 1113, 2019, 19930426, 19491001]
# seed_ls = [1, 67789, 83920001, 20200202, 7381910, 987, 92029, 18273, 29191, 5362]
# seed_ls = [567, 3412, 9090, 20148, 191901, 90901, 782716, 9101919, 19181918, 27181910]
# for seed in seed_ls:
#     np.random.seed(seed)
#  #     fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_30/3.shape/shape.c200T2M0m0.out.windowsHasNull/random_null/windowLen100.sliding100.valid_both.selflabel.train_truenull.txt'
#     fragment = '/home/gongjing/project/shape_imputation/data/hek_np_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/random_null/windowLen100.sliding100.train_truenull.txt'
#     savefn=fragment.replace('.txt', '.random0.1.s{}.txt'.format(seed))
#     util.data_random_null(fragment, null_pct=0.1, col=9, savefn=savefn)


# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen50.sliding50.txt'
# util.fragment_to_format_data(fragment=fragment, fragment_len=50, split=1, dataset='train')


# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.txt'
# util.fragment_split(fragment=fragment, train_frac=0.8)

out = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out'
# generate_data_set.generate_windows(out=out,  window_len_ls=[100], sliding_ls=[10], species='human', all_valid_reactivity=1, null_pct_max=0.9)
# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding10.txt'
# np.random.seed(1234)
# fragment_train,fragment_validate = util.fragment_split(fragment=fragment, train_frac=0.7, cols=8)
# util.data_random_null(fragment_train, null_pct=0.3, col=9, savefn=None)
# util.data_random_null(fragment_validate, null_pct=0.3, col=9, savefn=None)

# util.data_random_null('/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding10.blastn.validation.txt', null_pct=0.3, col=8, savefn=None)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.txt'
# util.fragment_split(fragment=fragment, train_frac=0.7, cols=9)
# util.data_random_null('/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.train_truenull.txt', null_pct=0.2, col=9)
# util.data_random_null('/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.validation_truenull.txt', null_pct=0.2, col=9)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.allvalid.txt'
# util.fragment_split(fragment=fragment, train_frac=0.7, cols=9)
# util.data_random_null('/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.allvalid.train_truenull.txt', null_pct=0.1, col=9)
# util.data_random_null('/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.allvalid.validation_truenull.txt', null_pct=0.1, col=9)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.txt'
# util.fragment_split(fragment=fragment, train_frac=0.7, cols=9)
# util.data_random_null('/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.train_truenull.txt', null_pct=0.1, col=9)
# util.data_random_null('/home/gongjing/project/shape_imputation/data/hek_wc_vivo_50/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.valid_both.validation_truenull.txt', null_pct=0.1, col=9)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_random/windowLen100.sliding100.train.txt'
# np.random.seed(1234)
# savefn=fragment.replace('.txt', '.{}.txt'.format(1234))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(9999)
# savefn=fragment.replace('.txt', '.{}.txt'.format(9999))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(5678)
# savefn=fragment.replace('.txt', '.{}.txt'.format(5678))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(12315)
# savefn=fragment.replace('.txt', '.{}.txt'.format(12315))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(400100)
# savefn=fragment.replace('.txt', '.{}.txt'.format(400100))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(42)
# savefn=fragment.replace('.txt', '.{}.txt'.format(42))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(1113)
# savefn=fragment.replace('.txt', '.{}.txt'.format(1113))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(2019)
# savefn=fragment.replace('.txt', '.{}.txt'.format(2019))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(19930426)
# savefn=fragment.replace('.txt', '.{}.txt'.format(19930426))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)
# np.random.seed(19491001)
# savefn=fragment.replace('.txt', '.{}.txt'.format(19491001))
# util.data_random_null(fragment, null_pct=0.1, col=8, savefn=savefn)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.txt'
# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.txt'
# np.random.seed(1234)
# savefn=fragment.replace('.txt', '.1perfragmentL{}.S{}.txt'.format(5, 1234))
# util.data_random_nullfragament(fragment, null_pct=0.1, col=8, savefn=savefn, mode='1perfragment', null_len=5, window_len=100)
# seed_ls = [1234, 9999, 5678, 12315, 400100, 42, 1113, 2019, 19930426, 19491001]
# null_len_ls = [5,10,15,20]
# for seed in seed_ls:
#     for null_len in null_len_ls:
#         np.random.seed(seed)
#         savefn=fragment.replace('.txt', '.1perfragmentL{}.S{}.txt'.format(null_len, seed))
#         util.data_random_nullfragament(fragment, null_pct=0.1, col=8, savefn=savefn, mode='1perfragment', null_len=null_len, window_len=100)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/random_null/windowLen100.sliding100.validation_truenull.txt'
# fragment = '/home/gongjing/project/shape_imputation/data/hek_cy_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/random_null/windowLen100.sliding100.train_truenull.txt'

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding10.random/windowLen100.sliding10.blastn.validation.txt'
# seed_ls = [1234,9999, 5678, 12315, 400100, 42, 1113, 2019, 19930426, 19491001]
# max_null_len_ls = [20]#[10, 20, 40]# [10, 20]#
# null_pct_ls = [0.3]#[0.3, 0.2, 0.1]#[0.3, 0.2, 0.1]
# for seed in seed_ls:
#     for max_null_len in max_null_len_ls:
#         for null_pct in null_pct_ls:
#             np.random.seed(seed)
#             savefn=fragment.replace('.txt', '.randomNperfragmentNullPct{}.maxL{}.S{}.txt'.format(null_pct, max_null_len, seed))
#             util.data_random_nullfragament(fragment, null_pct=null_pct, col=9, savefn=savefn, mode='randomNperfragment', null_len=5, window_len=100, max_null_len=max_null_len)
        
# sample_ls = ['22_trainLossall', '23_trainLossall_Gmultiply', '24_trainLossall_biLSTM', '25_trainLossall_biLSTMHid256', '26_trainLossall_biLSTMLay3', '27_trainLossall_biLSTMHid256Lay3', '28_trainLossall_LR0001', '29_trainLossall_LR0001_rep2']
# sample_ls = ['40_trainLossall_GmultiplyX_noise2']
# sample_ls = ['35_trainLossall_GmultiplyX_channel','38_trainLossall_GmultiplyX_biLSTMHid256Lay3','37_trainLossall_GmultiplyX_biLSTMLay3','36_trainLossall_GmultiplyX_biLSTMHid256','32_trainLossall_GmultiplyX_meswc']
# for sample in sample_ls:
#     log = '/home/gongjing/project/shape_imputation/exper/{}/log.txt'.format(sample)
#     util.read_log(log=log, savefn=log.replace('log.txt', 'loss.pdf'))


# generate noise data, not include original batch
# fragment = '/home/gongjing/project/shape_imputation/data/mes_wc_vivo/3.shape/c200T2/w100s100.train_null0.1.txt'
# savefn = fragment.replace('.txt','.noise2.txt')
# util.data_add_noise(fragment=fragment, ratio=2, col=7, seed=1234, savefn=savefn, noise=0.05)

# savefn = fragment.replace('.txt','.noise3.txt')
# util.data_add_noise(fragment=fragment, ratio=3, col=7, seed=1234, savefn=savefn, noise=0.05)

# savefn = fragment.replace('.txt','.noise4.txt')
# util.data_add_noise(fragment=fragment, ratio=4, col=7, seed=1234, savefn=savefn, noise=0.05)

# savefn = fragment.replace('.txt','.noise5.txt')
# util.data_add_noise(fragment=fragment, ratio=5, col=7, seed=1234, savefn=savefn, noise=0.05)

# savefn = fragment.replace('.txt','.noise10.txt')
# util.data_add_noise(fragment=fragment, ratio=10, col=7, seed=1234, savefn=savefn, noise=0.05)

# savefn = fragment.replace('.txt','.0.1noise2.txt')
# util.data_add_noise(fragment=fragment, ratio=2, col=7, seed=1234, savefn=savefn, noise=0.1)

# savefn = fragment.replace('.txt','.0.1noise5.txt')
# util.data_add_noise(fragment=fragment, ratio=5, col=7, seed=1234, savefn=savefn, noise=0.1)

# savefn = fragment.replace('.txt','.0.1noise10.txt')
# util.data_add_noise(fragment=fragment, ratio=10, col=7, seed=1234, savefn=savefn, noise=0.1)

# get .fa
# bed = '/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.bed'
# util.bed_get_fa(bed=bed, species='human')
# # search with fimo
# # /home/gongjing/software/meme_4.12.0/bin/fimo -oc ./test --thresh 0.05 --norc ./motif/Collapsed.used.meme ./human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.bed.fa
# motif_meme='/home/gongjing/project/shape_imputation/data/CLIP/motif/Collapsed.used.meme'
# fimo_dir=bed.replace('.bed', '.fimo')
# fimo='/home/gongjing/software/meme_4.12.0/bin/fimo'
# subprocess.call(["{} -oc {} --thresh 0.05 --norc {} {}".format(fimo, fimo_dir, motif_meme, bed.replace('.bed', '.fa'))], shell=True)
# util.fimo_convert('{}/fimo.txt'.format(fimo_dir))
# util.bed_fimo(bed=bed, species='human')

# bed = '/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.bed'
# util.bed_fimo(bed=bed, species='human')

# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20121_FXR1_HEK293_trx.tx_has_shape_region_null_ok.bed', species='human', motif='FXR')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20121_FXR1_HEK293_trx.tx_has_shape_region_null_exceed.bed', species='human', motif='FXR')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20122_FXR1_HEK293_trx.tx_has_shape_region_null_ok.bed', species='human', motif='FXR')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20122_FXR1_HEK293_trx.tx_has_shape_region_null_exceed.bed', species='human', motif='FXR')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20124_FXR2_HEK293_trx.tx_has_shape_region_null_ok.bed', species='human', motif='FXR') 
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20124_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.bed', species='human', motif='FXR')

# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20173_LIN28A_HEK293_trx.tx_has_shape_region_null_ok.bed', species='human', motif='LIN28')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20173_LIN28A_HEK293_trx.tx_has_shape_region_null_exceed.bed', species='human', motif='LIN28')

# IGFBP1
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20161_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.bed', species='human', motif='IGF2BP1_11')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_ok.bed', species='human', motif='IGF2BP1_11')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20161_IGF2BP1_HEK293_trx.tx_has_shape_region_null_exceed.bed', species='human', motif='IGF2BP1_11')
# util.bed_fimo(bed='/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20162_IGF2BP1_HEK293_trx.tx_has_shape_region_null_exceed.bed', species='human', motif='IGF2BP1_11')


# bed = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6AOld_site.tran.tx_has_shape_base_valid.bed'
# fa0 = util.bed_get_fa(bed=bed, species='human', extend=0, write_new_bed=1)
# fa5 = util.bed_get_fa(bed=bed, species='human', extend=5, write_new_bed=1)
# fa10 = util.bed_get_fa(bed=bed, species='human', extend=10, write_new_bed=1)
# fa50 = util.bed_get_fa(bed=bed, species='human', extend=50, write_new_bed=1)
# subprocess.call(["/home/gongjing/.local/bin/weblogo -f {} -D fasta -o {} -F pdf".format(fa0, fa0.replace('.fa', '.pdf'))], shell=True)
# subprocess.call(["/home/gongjing/.local/bin/weblogo -f {} -D fasta -o {} -F pdf".format(fa5, fa5.replace('.fa', '.pdf'))], shell=True)
# subprocess.call(["/home/gongjing/.local/bin/weblogo -f {} -D fasta -o {} -F pdf".format(fa10, fa10.replace('.fa', '.pdf'))], shell=True)
# subprocess.call(["/home/gongjing/.local/bin/weblogo -f {} -D fasta -o {} -F pdf".format(fa50, fa50.replace('.fa', '.pdf'))], shell=True)

# fragment = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_valid.e10.bed.shape100.txt'
# util.flter_null_fragment(fragment=fragment, col=7, null_pct=0.2)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_random/windowLen100.sliding100.train.txt'
# np.random.seed(1234)
# savefn = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_evenrandom/windowLen100.sliding100.train.txt'
# util.data_null_even_in_interval(fragment=fragment, null_pct=0.1, col=8, savefn=savefn)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_random/windowLen100.sliding100.train.txt'
# np.random.seed(1234)
# savefn = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.1perfragment_even.S{}.txt'.format(1234)
# util.data_random_nullfragament(fragment, null_pct=0.1, col=8, savefn=savefn, mode='1perfragment_even', null_len=5, window_len=100, max_null_len=10)

# seed_ls = [1234, 9999, 5678, 12315, 400100, 42, 1113, 2019, 19930426, 19491001]
# null_len_ls = [20]#[5,10,15,20]
# for seed in seed_ls:
#     for null_len in null_len_ls:
#         fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_random/windowLen100.sliding100.train.txt'
#         np.random.seed(seed)
#         savefn = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.1perfragment_evenL{}.S{}.txt'.format(null_len, seed)
#         util.data_random_nullfragament(fragment, null_pct=0.1, col=8, savefn=savefn, mode='1perfragment_even', null_len=null_len, window_len=100, max_null_len=10)

# seed_ls = [1234, 9999, 5678, 12315, 400100, 42, 1113, 2019, 19930426, 19491001]
# for seed in seed_ls:
#     fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_random/windowLen100.sliding100.train.txt'
#     savefn = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_evenrandom/windowLen100.sliding100.train.pct0.15.s{}.txt'.format(seed)
#     util.data_null_even_in_interval(fragment=fragment, null_pct=0.15, col=8, savefn=savefn, seed=seed)

# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_evenrandom/windowLen100.sliding100.validation.txt'
# np.random.seed(1234)
# savefn = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_evenrandom/windowLen100.sliding100.validation.s1234.txt'
# util.data_null_even_in_interval(fragment=fragment, null_pct=0.1, col=8, savefn=savefn, seed=1234)


# fn = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_valid.e0.bed.sort.shape'
# savefn = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_valid.e0.bed.sort.shape.heatmap.pdf'
# util.plot_heatmap(fn=fn, savefn=savefn, value_col=3, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black')

# fn = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_null.e0.bed.sort.shape'
# savefn = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6A_site.tran.tx_has_shape_base_null.e0.bed.sort.shape.heatmap.pdf'
# util.plot_heatmap(fn=fn, savefn=savefn, value_col=3, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black')

# fn = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6AsearchNegative_site.tran.tx_has_shape_base_null.e0.bed.sort.shape'
# savefn = '/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204/RMBase_hg38_all_m6AsearchNegative_site.tran.tx_has_shape_base_null.e0.bed.sort.shape.heatmap.pdf'
# util.plot_heatmap(fn=fn, savefn=savefn, value_col=3, fig_size_x=10, fig_size_y=20, cmap='summer', facecolor='black')

### mes/hek293 cy/np/ch
# fragment = '/home/gongjing/project/shape_imputation/data/mes_wc_vitro/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.txt'
# np.random.seed(1234)
# fragment_train,fragment_validate = util.fragment_split(fragment=fragment, train_frac=0.7, cols=8)
# util.data_random_null(fragment_train, null_pct=0.1, col=9, savefn=None)
# util.data_random_null(fragment_validate, null_pct=0.1, col=9, savefn=None)

# sample_ls = ['mes_ch_vitro','mes_ch_vivo', 'mes_cy_vitro', 'mes_cy_vivo', 'mes_np_vitro', 'mes_np_vivo']
# sample_ls = ['hek_ch_vitro','hek_ch_vivo', 'hek_cy_vitro', 'hek_cy_vivo', 'hek_np_vitro', 'hek_np_vivo']
# sample_ls = ['hek_wc_vitro']
# species = 'human'
# for sample in sample_ls:
#     out = '/home/gongjing/project/shape_imputation/data/{}/3.shape/shape.c200T2M0m0.out'.format(sample)
#     generate_data_set.generate_windows(out=out,  window_len_ls=None, sliding_ls=None, species=species, all_valid_reactivity=1, null_pct_max=0.9)
#     fragment = '/home/gongjing/project/shape_imputation/data/{}/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.txt'.format(sample)
#     np.random.seed(1234)
#     fragment_train,fragment_validate = util.fragment_split(fragment=fragment, train_frac=0.7, cols=8)
#     util.data_random_null(fragment_train, null_pct=0.1, col=9, savefn=None)
#     util.data_random_null(fragment_validate, null_pct=0.1, col=9, savefn=None)
    

# out = '/home/gongjing/project/shape_imputation/data/CIRSseq/CIRSseq_mES.out'
# species = 'mouse(CIRSseq)'
# generate_data_set.generate_windows(out=out,  window_len_ls=None, sliding_ls=None, species=species, all_valid_reactivity=1, null_pct_max=0.9)
# fragment = '/home/gongjing/project/shape_imputation/data/CIRSseq/CIRSseq_mES.out.windowsHasNull/windowLen100.sliding100.txt'
# np.random.seed(1234)
# fragment_train,fragment_validate = util.fragment_split(fragment=fragment, train_frac=0.7, cols=8)
# util.data_random_null(fragment_train, null_pct=0.1, col=9, savefn=None)
# util.data_random_null(fragment_validate, null_pct=0.1, col=9, savefn=None)


# fragment = '/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/validation_randomnullfragment/windowLen100.sliding100.validation.txt'
# seed_ls = [1234, 9999, 5678, 12315, 400100, 42, 1113, 2019, 19930426, 19491001]
# for seed in seed_ls:
#    np.random.seed(seed)
#    savefn = fragment.replace('.txt', '.randomNullDist.S{}.txt'.format(seed))
#    util.fragment_random_based_on_dist(fragment=fragment, col=8, savefn=savefn)


icshape_fragment_pct_plus_exceed_pct2='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.nominus.predict/iteration1/allfragment.0.5+exceed0.5.txt2'
icshape_fragment_pct_plus_exceed_predict='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.nominus.predict/iteration1/allfragment.0.5+exceed0.5.txt2.predict'
icshape_fragment_pct_plus_exceed_predict_shapeout='/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/test_prediction/hek_wc.out.c80.newwithNULL.nominus.predict/iteration1/allfragment.0.5+exceed0.5.txt2.predict.out'
util.predict_to_shape(validation=icshape_fragment_pct_plus_exceed_pct2, predict=icshape_fragment_pct_plus_exceed_predict, shape_out=icshape_fragment_pct_plus_exceed_predict_shapeout)
