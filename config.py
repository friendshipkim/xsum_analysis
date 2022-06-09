import torch
from os.path import join

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

# model
# model_name = "facebook/bart-large-xsum"
# distilbart
model_name = "sshleifer/distilbart-xsum-12-6"

# file path
base_cache_dir = "/home/wk247/workspace/xsum_analysis/cache"
ptb_docs_dir = join(base_cache_dir, "ptb_docs")
gen_seqs_dir = join(base_cache_dir, "gen_seqs")
log_probs_dir = join(base_cache_dir, "log_probs")
ner_dir = join(base_cache_dir, "ner")

# document perturbation
document_ptb_methods = ["insert", "ner"]
## 1. insert
insert_num_options = [1, 2]
insert_1_options = ["top1", "random"]
insert_2_options = ["topbottom", "top2", "random"]
num_max_insert = 10
## 2. ner
ner_tagger = "trf"
filter_labels = ["PERSON", "FAC", "GPE", "NORP", "LOC","EVENT","LANGUAGE", "LAW", "ORG"]
pool_size_reduction_ratio = 0.1

# summary generation
summary_generation_methods = ["true", "beam", "topp", "topk"]
max_summary_length = 150
num_return_seqs_per_trial = 30  # for sampling
max_trial = 30  # for sampling

# log prob
mask_idx = -100