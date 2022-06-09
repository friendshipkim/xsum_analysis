#!/bin/bash
source /home/wk247/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

GEN_METHOD="topp"
NUM_SEQS=30
P=0.8

conda activate xsum_analysis
python ~/workspace/xsum_analysis/generate_summary.py \
--gen_method ${GEN_METHOD} --num_return_seqs ${NUM_SEQS} --p ${P}