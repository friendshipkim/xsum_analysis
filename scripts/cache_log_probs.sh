#!/bin/bash
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate xsum_analysis

help() {
    echo "cache_summary.sh [OPTIONS]"
    echo "    -h			help"
    echo "    -m ARG		model_name: facebook/bart-large-xsum | sshleifer/distilbart-xsum-12-6 (default: facebook/bart-large-xsum)"
    echo "    -g ARG		gen_method: true | beam | topk | topp (default: true)"
    echo "    -n ARG		num_return_seqs : int (default: 30)"
    echo "    -b ARG 		num_beams : if gen_method == beam (default: 30)"
    echo "    -k ARG 		k : if gen_method == topk (default: 100)"
    echo "    -p ARG 		p : if gen_method == topp (default: 0.8)"
    exit 0
}

model_name="facebook/bart-large-xsum"
gen_method="true"
num_return_seqs=30
num_beams=30
k=100
p=0.8

export model_name

while getopts "m:g:n:b:k:p:h" opt
do
  case $opt in
    m) model_name=$OPTARG;;
	g) gen_method=$OPTARG;;
    n) num_return_seqs=$OPTARG;;
    b) num_beams=$OPTARG;;
	k) k=$OPTARG;;
    p) p=$OPTARG;;
    h) help ;;
    ?) help ;;
  esac
done

cache_log_probs_original(){
    python ~/workspace/xsum_analysis/calculate_log_probs_original.py \
    --model_name $model_name \
    --gen_method $gen_method \
    --num_return_seqs $num_return_seqs \
    --num_beams $num_beams \
    --k $k \
    --p $p
}

cache_log_probs_insert(){
    python ~/workspace/xsum_analysis/calculate_log_probs_ptb.py \
    --model_name $model_name \
    --gen_method $gen_method \
    --num_return_seqs $num_return_seqs \
    --num_beams $num_beams \
    --k $k \
    --p $p \
    --ptb_method insert \
    --insert_num $1 \
    --insert_position $2
}

cache_log_probs_ner(){
    python ~/workspace/xsum_analysis/calculate_log_probs_ptb.py \
    --model_name $model_name \
    --gen_method $gen_method \
    --num_return_seqs $num_return_seqs \
    --num_beams $num_beams \
    --k $k \
    --p $p \
    --ptb_method ner
}

insert1_position_list="top1 random"
insert2_position_list="top2 topbottom random"

cache_log_probs_original

for insert_position in $insert1_position_list; do
    cache_log_probs_insert 1 $insert_position
done

for insert_position in $insert2_position_list; do
    cache_log_probs_insert 2 $insert_position
done

cache_log_probs_ner





