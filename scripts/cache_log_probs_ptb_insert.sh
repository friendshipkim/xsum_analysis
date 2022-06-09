#!/bin/bash
source /home/wk247/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

GEN_METHOD="beam"
NUM_SEQS=30

PTB_METHOD="insert"
NUM_INSERT="2"
INSERT_POSITION="top2"

conda activate xsum_analysis
python /home/wk247/workspace/xsum_analysis/calculate_log_probs_ptb.py \
--gen_method ${GEN_METHOD} --num_seqs ${NUM_SEQS} \
--ptb_method ${PTB_METHOD} --num_insert ${NUM_INSERT} --insert_position ${INSERT_POSITION}