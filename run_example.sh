set -u 

INPUT=test/A.txt
k1=3
k2=8
OUTDIR=test

if [[ $1 == 1 ]]
then
	python3.8 runNMTF.py --in_file ${INPUT} --k1 ${k1} --k2 ${k2} --out_dir ${OUTDIR} --verbose --cpu
else 
	python3.8 runNMTF.py --in_file ${INPUT} --k1 ${k1} --k2 ${k2} --out_dir ${OUTDIR} --verbose
fi
