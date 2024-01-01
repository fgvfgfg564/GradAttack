ROOTDIR=$(dirname $0)

export PYTHONPATH=.

for model in bmshj2018_hyperprior bmshj2018_factorized mbt2018 cheng2020_attn
do
    python ${ROOTDIR}/main.py ${model} -o ${ROOTDIR}/${model}
done