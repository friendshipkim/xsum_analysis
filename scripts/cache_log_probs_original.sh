#!/bin/bash
source /home/wk247/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

GEN_METHOD="beam"
NUM_SEQS=30

conda activate xsum_analysis
python /home/wk247/workspace/xsum_analysis/calculate_log_probs_original.py \
--gen_method ${GEN_METHOD} --num_seqs ${NUM_SEQS}
