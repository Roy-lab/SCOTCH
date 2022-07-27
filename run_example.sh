set -u 

INPUT=test/A.txt
k1=3
k2=8
lU=0
lV=0
OUTDIR=test/lU_${lU}_lV_${lV}

mkdir -p $OUTDIR

if [[ $1 == 1 ]]
then
	python3.8 runNMTF.py --in_file ${INPUT} --k1 ${k1} --k2 ${k2} --out_dir ${OUTDIR} --verbose --device cpu --lU ${lU} --lV ${lV}
else 
	python3.8 runNMTF.py --in_file ${INPUT} --k1 ${k1} --k2 ${k2} --out_dir ${OUTDIR} --verbose --device cuda --lU ${lU} --lV ${lV}
fi
