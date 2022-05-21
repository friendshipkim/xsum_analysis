#!/bin/bash
source /home/wk247/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate xsum_analysis
python ~/workspace/xsum_analysis/calculate_log_probs_ood_insert_topbottom.py
