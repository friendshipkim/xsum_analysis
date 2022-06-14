import torch
import logging
from os.path import join, expanduser

# logging
handlers = [logging.FileHandler("filename.log"), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    handlers=handlers,
)
gen_seqs_logger = logging.getLogger("gen_seqs")
log_probs_logger = logging.getLogger("log_probs")
log_interval = 100

# cuda
if torch.cuda.is_available():
    empty_gpu_ids = [
        i for i in range(torch.cuda.device_count()) if torch.cuda.memory_usage(i) == 0
    ]
    if len(empty_gpu_ids) != 0:
        device = torch.device(f"cuda:{empty_gpu_ids[0]}")
        logging.info(f">> Running on cuda:{empty_gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    logging.info(f">> Running on cpu")
else:
    device = torch.device("cpu")
    logging.info(f">> Running on cpu")

# seed
seed = 0

# model
# model_name = "facebook/bart-large-xsum"
model_name = "sshleifer/distilbart-xsum-12-6"
model_dir = model_name.split("/")[-1]
logging.info(f">> Model: {model_name}")

# file path
base_cache_dir = expanduser("~/workspace/xsum_analysis/cache_tmp")
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
filter_labels = ["PERSON", "FAC", "GPE", "NORP", "LOC", "EVENT", "LANGUAGE", "LAW", "ORG"]
pool_size_reduction_ratio = 0.1

# summary generation
summary_generation_methods = ["true", "beam", "topp", "topk"]
max_summary_length = 150
num_return_seqs_per_trial = 30  # for sampling
max_trial = 30  # for sampling

# log prob
mask_idx = -100
