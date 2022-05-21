import datasets
from xsum_dataset import XsumDataset
from generate_xsum_summary import load_summarization_model_and_tokenizer

import argparse
import random

from typing import List
import torch
from torch import nn

from tqdm import tqdm
from ner_utils import *


random.seed(0)
torch.random.seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
model_name = "facebook/bart-large-xsum"
model, tokenizer = load_summarization_model_and_tokenizer(model_name)

# load test datasets
xsum_data_raw = datasets.load_dataset("xsum")
xsum_test_data = XsumDataset(xsum_data_raw["test"])

# hyperparameters
cache_dir = "/home/wk247/workspace/xsum_analysis/cache/gen_summary"
save_cache_dir = "/home/wk247/workspace/xsum_analysis/cache/log_probs"
num_return_seqs = 30

# load sequences list
gen_sequences_list = load_from_cache_dir(
        f"beam_gen_sequences_{num_return_seqs}",
        cache_dir)

# criterion
criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def get_log_probs(logits, labels):
    # losses of sequences, shape: [num_return_seqs, max_seq_len]
    # loss is negative log probability
    seq_losses = criterion(logits.permute(0, 2, 1), labels)
    seq_losses_masked = seq_losses.masked_fill(seq_losses==0., torch.nan)  # mask 0 with nan to ignore padding
    
    # log probabilities of sequences, shape: [num_return_seqs]
    seq_logprobs = -seq_losses_masked.nansum(1)

    return seq_logprobs


original_log_probs_list = []
for data, gen_sequences in tqdm(zip(xsum_test_data.dataset, gen_sequences_list), total = len(gen_sequences_list)):
    original_id = data["id"]
    original_doc = data["document"]
    true_summary = data["true_summary"]

    # tokenize original and ood documents
    inputs = tokenizer(
        [original_doc],
        # max_length=1024,  # default is 1024 for 'facebook/bart-large-xsum'
        truncation=True,
        return_tensors="pt",
        padding=True,
    )

    original_doc_token_ids = inputs.input_ids[0].to(device)

    # remove sos token to make labels
    gen_sequences_clean = gen_sequences[:, 1:]
    gen_labels = gen_sequences_clean.masked_fill(
        gen_sequences_clean == tokenizer.pad_token_id, -100
    ).to(device)  # pad with -100

    # feed the documents to the model to get log probs
    with torch.no_grad():
        original_model_output = model(
            input_ids=original_doc_token_ids.repeat(num_return_seqs, 1),
            labels=gen_labels,
        )

        original_log_probs = get_log_probs(original_model_output.logits, gen_labels)
    

    assert num_return_seqs == original_log_probs.size(0)
    original_log_probs_list.append(original_log_probs.cpu())

save_to_cache_dir(
    original_log_probs_list, 
    "original_log_probs_list",
    save_cache_dir
)
