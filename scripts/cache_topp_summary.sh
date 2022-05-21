#!/bin/bash
source /home/wk247/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate xsum_analysis
python ~/workspace/xsum_analysis/topp_sampling.py --num_return_seqs 20
