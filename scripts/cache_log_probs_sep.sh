#!/bin/bash
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate xsum_analysis

insert1_position_list="top1 random"
insert2_position_list="top2 topbottom random"

model_name="facebook/bart-large-xsum"
gen_method="topk"
num_return_seqs=40
k=100

export model_name

cache_log_probs_original(){
    python ~/workspace/xsum_analysis/calculate_log_probs_original.py \
    --model_name $model_name \
    --gen_method $gen_method \
    --num_return_seqs $num_return_seqs \
    --k $k
}

cache_log_probs_insert(){
    python ~/workspace/xsum_analysis/calculate_log_probs_ptb.py \
    --model_name $model_name \
    --gen_method $gen_method \
    --num_return_seqs $num_return_seqs \
    --k $k \
    --ptb_method insert \
    --insert_num $1 \
    --insert_position $2
}

cache_log_probs_ner(){
    python ~/workspace/xsum_analysis/calculate_log_probs_ptb.py \
    --model_name $model_name \
    --gen_method $gen_method \
    --num_return_seqs $num_return_seqs \
    --k $k \
    --ptb_method ner
}

cache_log_probs_original

for insert_position in $insert1_position_list; do
    cache_log_probs_insert 1 $insert_position
done

for insert_position in $insert2_position_list; do
    cache_log_probs_insert 2 $insert_position
done

cache_log_probs_ner