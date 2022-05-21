#!/bin/bash
source /home/wk247/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate xsum_analysis
python ~/workspace/xsum_analysis/cache_xsum_ner.py
