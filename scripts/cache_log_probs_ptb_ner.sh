#!/bin/bash
source /home/wk247/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

GEN_METHOD="true"
NUM_SEQS=1

PTB_METHOD="ner"

conda activate xsum_analysis
python /home/wk247/workspace/xsum_analysis/calculate_log_probs_ptb.py \
--gen_method ${GEN_METHOD} --num_seqs ${NUM_SEQS} \
--ptb_method ${PTB_METHOD}