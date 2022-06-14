import torch
from os.path import join

# gpu device
if torch.cuda.is_available():
    empty_gpu_id = [i for i in range(torch.cuda.device_count()) if torch.cuda.memory_usage(i)==0][0]
    device = torch.device(f"cuda:{empty_gpu_id}")
    print(f"running on cuda:{empty_gpu_id}")
else: 
    device = torch.device("cpu")
    print(f"running on cpu")

seed = 0

# model
# model_name = "facebook/bart-large-xsum"

# distilled model
model_name = "sshleifer/distilbart-xsum-12-6"

model_dir = model_name.split("/")[-1]

# file path
base_cache_dir = "~/workspace/xsum_analysis/cache"
ptb_docs_dir = join(base_cache_dir, "ptb_docs")
gen_seqs_dir = join(base_cache_dir, "gen_seqs", model_dir)
log_probs_dir = join(base_cache_dir, "log_probs", model_dir)
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
num_return_seqs_per_trial = 50  # for sampling
max_trial = 30  # for sampling

# log prob
mask_idx = -100
