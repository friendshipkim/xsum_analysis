"""
Script to run inference (inputs: original document, labels: generated summaries) 
and save log probabilities of the generated summaries
This can be merged if log probs can be calculated while generating sequences
"""

import datasets
from xsum_dataset import XsumDataset
from generate_xsum_summary import load_summarization_model_and_tokenizer

import argparse
import random

# from typing import List
import torch
from torch import nn, Tensor

from tqdm import tqdm
from ner_utils import *


random.seed(0)
torch.random.seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Given logit tensor and labels, calculate log probs of each sequences
    Args:
        logits: logit tensor (shape: [num_seqs, max_seq_len, vocab_size])
        labels: label tensor (shape: [num_seqs, max_seq_len])
    Returns:
        seq_logprobs: torch.Tensor (shape: [num_seqs])
    """
    # losses of sequences, shape: [num_seqs, max_seq_len]
    # loss is negative log probability
    seq_losses = criterion(logits.permute(0, 2, 1), labels)
    seq_losses_masked = seq_losses.masked_fill(seq_losses==0., torch.nan)  # mask 0 with nan to ignore padding
    
    # log probabilities of sequences, shape: [num_seqs]
    seq_logprobs = -seq_losses_masked.nansum(1)

    return seq_logprobs


# ========== default parameters ==========
model_name = "facebook/bart-large-xsum"
gen_sum_dir = "/home/wk247/workspace/xsum_analysis/cache/gen_summary"
log_probs_dir = "/home/wk247/workspace/xsum_analysis/cache/log_probs"

summary_generation_methods = ["beam", "topp", "topk"]
# ========================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run inference on the generated summaries and save log probabilities of them"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=model_name,
        help="Summarization model to test (Default: facebook/bart-large-xsum)",
    )

    parser.add_argument(
        "--gen_method",
        type=str,
        required=True,
        choices=summary_generation_methods,
        help=f"Method used to generate summaries (Choices: {summary_generation_methods})",
    )

    parser.add_argument(
        "--num_seqs",
        type=int,
        required=True,
        help="Number of generated summaries",
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # load model and tokenizer
    model, tokenizer = load_summarization_model_and_tokenizer(args.model_name)

    # load test dataset
    xsum_data_raw = datasets.load_dataset("xsum")
    xsum_test_dataset = XsumDataset(xsum_data_raw["test"]).dataset

    # load sequences list
    gen_seqs_list = load_from_cache_dir(
            f"{args.gen_method}_gen_sequences_{args.num_seqs}",
            gen_sum_dir)

    # criterion
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    # ======= calaulate log probs of each sequences
    original_log_probs_list = []
    for data, gen_seqs in tqdm(zip(xsum_test_dataset, gen_seqs_list), total = len(gen_seqs_list)):
        original_id = data["id"]
        original_doc = data["document"]
        true_summary = data["true_summary"]

        # tokenize original document
        inputs = tokenizer(
            original_doc,
            # max_length=1024,  # default is 1024 for 'facebook/bart-large-xsum'
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        original_doc_token_ids = inputs.input_ids.to(device)

        # process generated sequences to use them as labels
        gen_sequences_clean = gen_seqs[:, 1:]  # remove <sos> token to make labels
        gen_labels = gen_sequences_clean.masked_fill(
            gen_sequences_clean == tokenizer.pad_token_id, -100
        ).to(device)  # pad with -100

        # feed the document to the model to get log probs
        with torch.no_grad():
            original_model_output = model(
                input_ids=original_doc_token_ids.repeat(args.num_seqs, 1),
                labels=gen_labels,
            )
        original_log_probs = calculate_log_probs(original_model_output.logits, gen_labels).cpu()
        
        # append to list
        assert args.num_seqs == original_log_probs.size(0)
        original_log_probs_list.append(original_log_probs)

    assert len(original_log_probs_list) == len(xsum_test_dataset)

    # save it to cache dir
    save_to_cache_dir(
        original_log_probs_list,
        f"original_log_probs_list_{args.gen_method}_{args.num_seqs}",
        log_probs_dir
    )