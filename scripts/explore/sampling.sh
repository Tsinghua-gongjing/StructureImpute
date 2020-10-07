sample=hek_wc_vivo
sample_dir=/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data
D1=$sample_dir/$sample/0.rawData/dmso_1.trimmed.fastq
D2=$sample_dir/$sample/0.rawData/dmso_2.trimmed.fastq
N1=$sample_dir/$sample/0.rawData/nai_1.trimmed.fastq
N2=$sample_dir/$sample/0.rawData/nai_2.trimmed.fastq

# for i in 10 20 30 40 50 60 70 
# do
# 	mkdir -p $sample_dir/${sample}_${i}/0.rawData
# 	dmso_reads_sample=$((1000000*$i))
# 	nai_reads_sample=$((1000000*$i*2))
# 	echo "sampling",$i,$dmso_reads_sample,$nai_reads_sample
# 	seqtk sample -s 1234 $D1 $dmso_reads_sample > $sample_dir/${sample}_${i}/0.rawData/dmso_1.trimmed.fastq
# 	seqtk sample -s 1234 $D2 $dmso_reads_sample > $sample_dir/${sample}_${i}/0.rawData/dmso_2.trimmed.fastq
# 	seqtk sample -s 1234 $D1 $nai_reads_sample > $sample_dir/${sample}_${i}/0.rawData/nai_1.trimmed.fastq
# 	seqtk sample -s 1234 $D2 $nai_reads_sample > $sample_dir/${sample}_${i}/0.rawData/nai_2.trimmed.fastq
# done


# for i in 10 20 30 40 50 60 70
# do
# 	bsub -q Z-ZQF -eo sampling${i}.err -oo sampling${i}.out bash icSHAPE-pipe.sh hek_wc_vivo_${i} human
# 	sleep 20
# done


# for i in 30 50 60 
# do
# 	for seed in 1234 40 9988 17181790 81910 625178 1 7829999 9029102 918029109
# 	do
# 		mkdir -p $sample_dir/sampling/${sample}_${i}_${seed}/0.rawData
# 		dmso_reads_sample=$((1000000*$i))
# 		nai_reads_sample=$((1000000*$i*2))
# 		echo "sampling",$i,$dmso_reads_sample,$nai_reads_sample
# 		seqtk sample -s $seed $D1 $dmso_reads_sample > $sample_dir/sampling/${sample}_${i}_${seed}/0.rawData/dmso_1.trimmed.fastq
# 		seqtk sample -s $seed $D2 $dmso_reads_sample > $sample_dir/sampling/${sample}_${i}_${seed}/0.rawData/dmso_2.trimmed.fastq
# 		seqtk sample -s $seed $D1 $nai_reads_sample > $sample_dir/sampling/${sample}_${i}_${seed}/0.rawData/nai_1.trimmed.fastq
# 		seqtk sample -s $seed $D2 $nai_reads_sample > $sample_dir/sampling/${sample}_${i}_${seed}/0.rawData/nai_2.trimmed.fastq
# 	done
# done

for i in 30 50 60 
do
	for seed in 1234 40 9988 17181790 81910 625178 1 7829999 9029102 918029109
	do
		bsub -q Z-ZQF -eo sampling${i}.s${seed}.err -oo sampling${i}.s${seed}.out bash icSHAPE-pipe.sh sampling/${sample}_${i}_${seed} human
		sleep 20
	done
done

# depth_sampling(){

# 		depth=$1
# 		seed=$2

# 		sample=hek_wc_vivo
# 		sample_dir=/Share2/home/zhangqf5/gongjing/RNA-structure-profile-imputation/data
# 		D1=$sample_dir/$sample/0.rawData/dmso_1.trimmed.fastq
# 		D2=$sample_dir/$sample/0.rawData/dmso_2.trimmed.fastq
# 		N1=$sample_dir/$sample/0.rawData/nai_1.trimmed.fastq
# 		N2=$sample_dir/$sample/0.rawData/nai_2.trimmed.fastq

# 		mkdir -p $sample_dir/sampling/${sample}_${depth}_${seed}/0.rawData
# 		dmso_reads_sample=$((1000000*$i))
# 		nai_reads_sample=$((1000000*$i*2))
# 		echo "sampling",$i,$dmso_reads_sample,$nai_reads_sample
# 		seqtk sample -s $seed $D1 $dmso_reads_sample > $sample_dir/sampling/${sample}_${depth}_${seed}/0.rawData/dmso_1.trimmed.fastq
# 		seqtk sample -s $seed $D2 $dmso_reads_sample > $sample_dir/sampling/${sample}_${depth}_${seed}/0.rawData/dmso_2.trimmed.fastq
# 		seqtk sample -s $seed $D1 $nai_reads_sample > $sample_dir/sampling/${sample}_${depth}_${seed}/0.rawData/nai_1.trimmed.fastq
# 		seqtk sample -s $seed $D2 $nai_reads_sample > $sample_dir/sampling/${sample}_${depth}_${seed}/0.rawData/nai_2.trimmed.fastq
# }




