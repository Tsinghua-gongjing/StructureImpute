export PATH=/Share/home/zhangqf/lipan/usr/icSHAPE-pipe/bin:$PATH

project_dir=/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation

sample=hek_ch_vivo
data_dir=$project_dir/data/$sample
species=human

case $species in
	human)
		ref_fa=$project_dir/data/ref/hg38/hg38_transcriptome.fa
		star_index=$project_dir/data/ref/hg38
		;;
	mouse)
		ref_fa=$project_dir/data/ref/mm10/mm10_transcriptome.fa
		star_index=$project_dir/data/ref/mm10
		;;
esac



######################
# build index
######################
# icSHAPE-pipe starbuild -i $ref_fa -o /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/mm10/
# icSHAPE-pipe starbuild -i $ref_fa -o /Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data/ref/hg38/



######################
# pre-processing
######################

# bsub -q Z-ZQF -eo $data_dir/0.rawData/readcollapse.dmso1.err -oo $data_dir/0.rawData/readcollapse.dmso1.out "icSHAPE-pipe readcollapse -U $data_dir/0.rawData/dmso_1.fastq.gz -o $data_dir/0.rawData/dmso_1.rmdup.fastq --simplify"
# bsub -q Z-ZQF -eo $data_dir/0.rawData/readcollapse.dmso2.err -oo $data_dir/0.rawData/readcollapse.dmso2.out "icSHAPE-pipe readcollapse -U $data_dir/0.rawData/dmso_2.fastq.gz -o $data_dir/0.rawData/dmso_2.rmdup.fastq --simplify"
# bsub -q Z-ZQF -eo $data_dir/0.rawData/readcollapse.nai1.err -oo $data_dir/0.rawData/readcollapse.nai1.out "icSHAPE-pipe readcollapse -U $data_dir/0.rawData/nai_1.fastq.gz -o $data_dir/0.rawData/nai_1.rmdup.fastq --simplify"
# bsub -q Z-ZQF -eo $data_dir/0.rawData/readcollapse.nai2.err -oo $data_dir/0.rawData/readcollapse.nai2.out "icSHAPE-pipe readcollapse -U $data_dir/0.rawData/nai_2.fastq.gz -o $data_dir/0.rawData/nai_2.rmdup.fastq --simplify"

# adaptor=/Share2/home/zhangqf5/gongjing/Kethoxal_RNA_structure/scripts/icSHAPE-master/data/adapter/TruSeq3-SE-add.fa
# bsub -q Z-ZQF -eo $data_dir/0.rawData/trim.dmso1.err -oo $data_dir/0.rawData/trim.dmso1.out "icSHAPE-pipe trim -i $data_dir/0.rawData/dmso_1.rmdup.fastq -o $data_dir/0.rawData/dmso_1.trimmed.fastq -l 13 -a $adaptor"
# bsub -q Z-ZQF -eo $data_dir/0.rawData/trim.dmso2.err -oo $data_dir/0.rawData/trim.dmso2.out "icSHAPE-pipe trim -i $data_dir/0.rawData/dmso_2.rmdup.fastq -o $data_dir/0.rawData/dmso_2.trimmed.fastq -l 13 -a $adaptor"
# bsub -q Z-ZQF -eo $data_dir/0.rawData/trim.nai1.err -oo $data_dir/0.rawData/trim.nai1.out "icSHAPE-pipe trim -i $data_dir/0.rawData/nai_1.rmdup.fastq -o $data_dir/0.rawData/nai_1.trimmed.fastq -l 13 -a $adaptor"
# bsub -q Z-ZQF -eo $data_dir/0.rawData/trim.nai2.err -oo $data_dir/0.rawData/trim.nai2.out "icSHAPE-pipe trim -i $data_dir/0.rawData/nai_2.rmdup.fastq -o $data_dir/0.rawData/nai_2.trimmed.fastq -l 13 -a $adaptor"

# trim 3'len: 15 for mES ch/cp/cy, 13 for mES wc, hek wc/ch/np/cy


######################
# map to transcriptome
######################

# D1=$data_dir/0.rawData/dmso_1.trimmed.fastq
# D2=$data_dir/0.rawData/dmso_2.trimmed.fastq
# N1=$data_dir/0.rawData/nai_1.trimmed.fastq
# N2=$data_dir/0.rawData/nai_2.trimmed.fastq

# mkdir -p $data_dir/1.mapGenome
# bsub -q Z-ZQF -eo $data_dir/1.mapGenome/map.dmso1.err -oo $data_dir/1.mapGenome/map.dmso1.out "icSHAPE-pipe mapGenome -i $D1 -o $data_dir/1.mapGenome/D1 -x $star_index -p 16 --maxMMap 10"
# bsub -q Z-ZQF -eo $data_dir/1.mapGenome/map.dmso1.err -oo $data_dir/1.mapGenome/map.dmso1.out "icSHAPE-pipe mapGenome -i $D2 -o $data_dir/1.mapGenome/D2 -x $star_index -p 16 --maxMMap 10"
# bsub -q Z-ZQF -eo $data_dir/1.mapGenome/map.dmso1.err -oo $data_dir/1.mapGenome/map.dmso1.out "icSHAPE-pipe mapGenome -i $N1 -o $data_dir/1.mapGenome/N1 -x $star_index -p 16 --maxMMap 10"
# bsub -q Z-ZQF -eo $data_dir/1.mapGenome/map.dmso1.err -oo $data_dir/1.mapGenome/map.dmso1.out "icSHAPE-pipe mapGenome -i $N2 -o $data_dir/1.mapGenome/N2 -x $star_index -p 16 --maxMMap 10"



######################
# sam to tab
######################

mkdir -p $data_dir/2.tab
icSHAPE-pipe sam2tab -in $data_dir/1.mapGenome/D1.sorted.bam -out $data_dir/2.tab/D1.tab
icSHAPE-pipe sam2tab -in $data_dir/1.mapGenome/D2.sorted.bam -out $data_dir/2.tab/D2.tab
icSHAPE-pipe sam2tab -in $data_dir/1.mapGenome/N1.sorted.bam -out $data_dir/2.tab/N1.tab
icSHAPE-pipe sam2tab -in $data_dir/1.mapGenome/N2.sorted.bam -out $data_dir/2.tab/N2.tab


# ######################
# # calc rpkm
# ######################
mkdir -p $data_dir/4.rpkm
samtools view -h -o $data_dir/1.mapGenome/D1.sorted.sam $data_dir/1.mapGenome/D1.sorted.bam
samtools view -h -o $data_dir/1.mapGenome/D2.sorted.sam $data_dir/1.mapGenome/D2.sorted.bam
samtools view -h -o $data_dir/1.mapGenome/N1.sorted.sam $data_dir/1.mapGenome/N1.sorted.bam
samtools view -h -o $data_dir/1.mapGenome/N2.sorted.sam $data_dir/1.mapGenome/N2.sorted.bam
calc_rpkm=/Share2/home/zhangqf5/gongjing/Kethoxal_RNA_structure/scripts/icSHAPE-master/scripts/estimateRPKM.pl
[ -e $data_dir/4.rpkm/D1.rpkm ] && rm $data_dir/4.rpkm/D1.rpkm ; perl $calc_rpkm -i $data_dir/1.mapGenome/D1.sorted.sam -o $data_dir/4.rpkm/D1.rpkm
[ -e $data_dir/4.rpkm/D2.rpkm ] && rm $data_dir/4.rpkm/D2.rpkm ; perl $calc_rpkm -i $data_dir/1.mapGenome/D2.sorted.sam -o $data_dir/4.rpkm/D2.rpkm
[ -e $data_dir/4.rpkm/N1.rpkm ] && rm $data_dir/4.rpkm/N1.rpkm ; perl $calc_rpkm -i $data_dir/1.mapGenome/N1.sorted.sam -o $data_dir/4.rpkm/N1.rpkm
[ -e $data_dir/4.rpkm/N2.rpkm ] && rm $data_dir/4.rpkm/N2.rpkm ; perl $calc_rpkm -i $data_dir/1.mapGenome/N2.sorted.sam -o $data_dir/4.rpkm/N2.rpkm

# ######################
# # calc shape value
# ######################

mkdir -p $data_dir/3.shape
icSHAPE-pipe calcSHAPE \
			-D $data_dir/2.tab/D1.tab,$data_dir/2.tab/D2.tab \
			-N $data_dir/2.tab/N1.tab,$data_dir/2.tab/N2.tab \
			-size $star_index/chrNameLength.txt \
			-out $data_dir/3.shape/output.tab\
			-genome $ref_fa \
			-bases A,T,C,G \
			-omc 0 \
			-mc 0 \
			-non-sliding

# icSHAPE-pipe genSHAPEToTransSHAPE \
# 	   -s $star_index/chrNameLength.txt \
# 	   -i $data_dir/3.shape/output.tab \
# 	   -o $data_dir/3.shape/shape.c0T0M0m0.out \
# 	   -c 0 \
# 	   -T 0	\
# 	   -M 0	\
# 	   -m 0

# icSHAPE-pipe genSHAPEToTransSHAPE \
# 	   -s $star_index/chrNameLength.txt \
# 	   -i $data_dir/3.shape/output.tab \
# 	   -o $data_dir/3.shape/shape.c200T2M0m0.out \
# 	   -c 100 \
# 	   -T 1	\
# 	   -M 0	\
# 	   -m 0

for c in 0 50 100 150 200 250
do
	for T in 0 1 2 3
	do
		echo $c,$T
		bsub -q Z-ZQF -eo $data_dir/3.shape/genSHAPEToTransSHAPE.c${c}T${T}.err -oo $data_dir/3.shape/genSHAPEToTransSHAPE.c${c}T${T}.out "icSHAPE-pipe genSHAPEToTransSHAPE -s $star_index/chrNameLength.txt -i $data_dir/3.shape/output.tab -o $data_dir/3.shape/shape.c${c}T${T}M0m0.out -c $c -T $T -M 0 -m 0"
	done
done