#!/bin/bash 

data_file=$1 #/mnt/dv/wid/projects6/Roy-singlecell3/sridharanlab2/data/perturb-seq_on_reprogramming/ke_work/NMTF_analysis_v2/input_data/A2S_conditions_with_MEF_Fibroblast_and_iPSC_merged_noheader_union.pt
regnet=$2 #/mnt/dv/wid/projects2/Roy-common/data/data_new/mouse/go/mm10_filt_level_3_4/mm10_gobp_regnet_3_4.txt
k1=$3
k2=$4
lambda_u=$5
lambda_v=$6
seed=$7

# Starting job
echo "==== Initializing env at $(date)"

outdir=k1_${k1}_k2_${k2}_lU_${lambda_u}_lV_${lambda_v}_seed_${seed}

mkdir -p $outdir

## Setup scotch env: 
export PATH=$PWD/bin:$PATH
tar -xzf SCOTCH.tar.gz
./bin/conda-unpack
# Activate the environment
source ./bin/activate

echo "==== Job started at $(date)"
echo "Running: " 
echo "python run_scotch.py --data_file $data_file --output_dir $outdir --regnet_file $regnet --k1 $k1 --k2 $k2 --lambda_u $lambda_u --lambda_v $lambda_v --device cpu --verbose --seed $seed --skip_visualization"

python run_scotch.py --data_file $data_file --output_dir $outdir --regnet_file $regnet --k1 $k1 --k2 $k2 --lambda_u $lambda_u --lambda_v $lambda_v --device cpu --verbose --seed $seed --skip_visualization > $outdir/scotch_log.txt 

echo "==== Job finished at $(date)"

# Compiling outdir 
tar -czf ${outdir}.tar.gz ${outdir}

echo "==== Completion at $(date)"
