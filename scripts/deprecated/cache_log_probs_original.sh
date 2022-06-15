#!/bin/bash
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

GEN_METHOD="beam"
NUM_RETURN_SEQS=30
NUM_BEAMS=30

conda activate xsum_analysis
python ~/workspace/xsum_analysis/calculate_log_probs_original.py \
--gen_method ${GEN_METHOD} --num_return_seqs ${NUM_RETURN_SEQS} --num_beams ${NUM_BEAMS}
