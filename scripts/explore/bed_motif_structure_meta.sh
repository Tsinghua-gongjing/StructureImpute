# python bed_motif_structure_meta.py /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt:/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.fimo/fimo.new.FXR.txt null:ok /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out:/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_ok.fimo/meta.pdf 10

# sample=CLIPDB20123_FXR2_HEK293_trx
# sample_dir=/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip
# null_exceed=$sample_dir/$sample.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt
# null_ok=$sample_dir/$sample.tx_has_shape_region_null_ok.fimo/fimo.new.FXR.txt
# shape=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out
# savefn=$sample_dir/$sample.tx_has_shape_region_null_ok.fimo/meta.pdf
# python bed_motif_structure_meta.py $null_exceed:$null_ok null:ok $shape:$shape $savefn 10

# CLIP
run_sample_meta(){
sample=$1
sample_dir=/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip
null_exceed=$sample_dir/$sample.tx_has_shape_region_null_exceed.fimo/fimo.new.$2.txt
null_ok=$sample_dir/$sample.tx_has_shape_region_null_ok.fimo/fimo.new.$2.txt
shape=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out
savefn=$sample_dir/$sample.tx_has_shape_region_null_ok.fimo/meta.RdYlGn_r.pdf
python bed_motif_structure_meta.py $null_exceed:$null_ok null:ok $shape:$shape $savefn 10

null_ok_bed_shape100_txt=$sample_dir/$sample.tx_has_shape_region_null_ok.fimo/fimo.new.$2.txt.shape100.txt
null_ok_bed_shape=$sample_dir/$sample.tx_has_shape_region_null_ok.fimo/fimo.new.$2.txt.shape
python sort_shape_by_null_count.py --shape1 $null_ok_bed_shape100_txt --shape2 $null_ok_bed_shape --value_col1 7
null_ok_bed_shape_sort=$sample_dir/$sample.tx_has_shape_region_null_ok.fimo/fimo.new.$2.txt.sort.shape
python plot_sample_heatmap.py --f $null_ok_bed_shape_sort --savefn $null_ok_bed_shape_sort.pdf --cmap RdYlGn_r --facecolor "#CBCBCB" --col 14 --null_type null_count

null_exceed_bed_shape100_txt=$sample_dir/$sample.tx_has_shape_region_null_exceed.fimo/fimo.new.$2.txt.shape100.txt
null_exceed_bed_shape=$sample_dir/$sample.tx_has_shape_region_null_exceed.fimo/fimo.new.$2.txt.shape
python sort_shape_by_null_count.py --shape1 $null_exceed_bed_shape100_txt --shape2 $null_exceed_bed_shape --value_col1 7
null_exceed_bed_shape_sort=$sample_dir/$sample.tx_has_shape_region_null_exceed.fimo/fimo.new.$2.txt.sort.shape
python plot_sample_heatmap.py --f $null_exceed_bed_shape_sort --savefn $null_exceed_bed_shape_sort.pdf --cmap RdYlGn_r --facecolor "#CBCBCB" --col 14 --null_type null_count
}

# run_sample_meta CLIPDB20121_FXR1_HEK293_trx 
# run_sample_meta CLIPDB20122_FXR1_HEK293_trx 
# run_sample_meta CLIPDB20123_FXR2_HEK293_trx FXR
# run_sample_meta CLIPDB20173_LIN28A_HEK293_trx LIN28

# run_sample_meta CLIPDB20162_IGF2BP1_HEK293_trx IGF2BP1_11
# run_sample_meta CLIPDB20161_IGF2BP1_HEK293_trx IGF2BP1_11
run_sample_meta CLIPDB20173_LIN28A_HEK293_trx LIN28


# RNA modification
run_sample_meta_modification(){
sample=$1
sample_dir=/home/gongjing/project/shape_imputation/data/RBMbase/download_20191204
null=$sample_dir/RMBase_hg38_all_$1_site.tran.tx_has_shape_base_null.e0.bed
valid=$sample_dir/RMBase_hg38_all_$1_site.tran.tx_has_shape_base_valid.e0.bed
shape=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out
shape=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/from_panpan/shape.out
savefn=$sample_dir/RMBase_hg38_all_$1_site.tran.tx_has_shape_base_null.e0.meta.RdYlGn_r.pdf
python bed_motif_structure_meta.py $null:$valid null:ok $shape:$shape $savefn $2
}

# run_sample_meta_modification m1A 10
# run_sample_meta_modification PseudoU 10
# run_sample_meta_modification m6A 10

# run_sample_meta_modification m1A 10
# run_sample_meta_modification PseudoU 10
# run_sample_meta_modification m6A 10

# run_sample_meta_modification m6Anegative 10
# run_sample_meta_modification m6Anegative 0

# run_sample_meta_modification m6AsearchNegative 10

# run_sample_meta_modification m6AOld 10
# run_sample_meta_modification m6AOldsearchNegative 10

### start/stop codon
start_codon_null=/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.start_codon.null.sort.shape100.bed
start_codon_valid=/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.start_codon.ok.sort.shape100.bed
shape=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out
savefn=/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.start_codon.sort.shape100.e0.meta.RdYlGn_r.pdf
# python bed_motif_structure_meta.py $start_codon_null:$start_codon_valid null:ok $shape:$shape $savefn 0

stop_codon_null=/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.stop_codon.null.sort.shape100.bed
stop_codon_valid=/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.stop_codon.ok.sort.shape100.bed
shape=/home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out
savefn=/home/gongjing/project/shape_imputation/results/start_stop_codon/hek_wc.stop_codon.sort.shape100.e0.meta.RdYlGn_r.pdf
# python bed_motif_structure_meta.py $stop_codon_null:$stop_codon_valid null:ok $shape:$shape $savefn 0