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



def decode_mult_seqs(
    seq_tokens: torch.LongTensor, skip_special_tokens: bool = True
) -> List[str]:
    return [
        tokenizer.decode(
            seq, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for seq in seq_tokens
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate possible summaries"
    )

    # parser.add_argument(
    #     "--bbc_id",
    #     type=int,
    #     required=True,
    #     help="A document BBC ID in the Xsum dataset",
    # )

    parser.add_argument(
        "--num_beams", 
        type=int, 
        required=False, 
        help="beam size",
    )

    parser.add_argument(
        "--num_return_seqs",
        type=int,
        required=True,
        help="the number of returned sequences (the size of summary pool)",
    )

    parser.add_argument(
        "--early_stopping",
        type=bool,
        required=False,
        default=True,
        help="whether to ealy stop the beam search (default: True))",
    )

    args = parser.parse_args()

    # randomly sample one bbcid
    # bbc_id = random.choice(list(xsum_test_data.data_by_id.keys()))
    # selected_data = xsum_test_data.data_by_id[bbc_id]

    topp_gen_sequences = []
    for i, data in enumerate(xsum_test_data.dataset):

        if i % 1000 == 0:
            print(f"{i} samples processed")
            
        original_doc = data["document"]
        true_summary = data["true_summary"]
        # ood_doc = selected_data["ood_document"]  # TODO

        # tokenize original and ood documents
        inputs = tokenizer(
            [original_doc], #, ood_doc],
            # max_length=1024,  # default is 1024 for 'facebook/bart-large-xsum'
            truncation=True,
            return_tensors="pt",
            padding=True,
        )

        original_doc_token_ids = inputs.input_ids[0].to(device)

        # ood_doc_token_ids = inputs.input_ids[1].to(device)
        # ood_attention_mask = inputs.attention_mask[1].to(device)

        # topp sampling
        max_length = 100

        sample_multi_output = model.generate(
            original_doc_token_ids[None, :],
            do_sample=True, 
            max_length=max_length, 
            top_p=0.9, 
            # top_k=30,
            num_return_sequences=args.num_return_seqs,
        )

        topp_gen_sequences.append(sample_multi_output.cpu())

    save_to_cache_dir(
        topp_gen_sequences, 
        f"topp_gen_sequences_{args.num_return_seqs}",
        cache_dir)

    exit()