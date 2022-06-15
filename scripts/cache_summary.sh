#!/bin/bash
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate xsum_analysis

cache_summary_true(){
python ~/workspace/xsum_analysis/generate_summary.py \
   --model_name $1 \
   --gen_method true \
   --num_return_seqs $2
}

cache_summary_beam(){
python ~/workspace/xsum_analysis/generate_summary.py \
   --model_name $1 \
   --gen_method beam \
   --num_return_seqs $2 \
   --num_beams $3
}

cache_summary_topk(){
python ~/workspace/xsum_analysis/generate_summary.py \
   --model_name $1 \
   --gen_method topk \
   --num_return_seqs $2 \
   --k $3
}

cache_summary_topp(){
python ~/workspace/xsum_analysis/generate_summary.py \
   --model_name $1 \
   --gen_method topp \
   --num_return_seqs $2 \
   --p $3
}

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

if [ $gen_method = "true" ]
then
    cache_summary_true $model_name $num_return_seqs
elif [ $gen_method = "beam" ]
then
    cache_summary_beam $model_name $num_return_seqs $num_beams
elif [ $gen_method = "topk" ]
then
    cache_summary_topk $model_name $num_return_seqs $k
elif [ $gen_method = "topp" ]
then
    cache_summary_topp $model_name $num_return_seqs $p
else
    echo "$gen_method is invalid"
fi