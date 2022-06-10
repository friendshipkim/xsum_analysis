#!/bin/bash
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate xsum_analysis
python ~/workspace/xsum_analysis/ptb/cache_xsum_ner_train.py
