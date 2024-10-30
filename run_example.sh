set -u 

INPUT=/mnt/dv/wid/projects5/Roy-singlecell/sr_work/multitask_matfact/datasets/liger/pbmc_alignment_srprocess/data/pbmc_10x_transpose.txt
k1=5
k2=5
lU=0
lV=0
OUTDIR=test/lU_${lU}_lV_${lV}

mkdir -p $OUTDIR

if [[ $1 == 1 ]]
then
	python runNMTF.py --in_file ${INPUT} --k1 ${k1} --k2 ${k2} --out_dir ${OUTDIR} --verbose --device cpu --lU ${lU} --lV ${lV}
else 
	python runNMTF.py --in_file ${INPUT} --k1 ${k1} --k2 ${k2} --out_dir ${OUTDIR} --verbose --device cuda --lU ${lU} --lV ${lV}
fi
