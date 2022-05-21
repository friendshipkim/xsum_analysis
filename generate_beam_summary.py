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

    beam_gen_sequences = []
    for data in tqdm(xsum_test_data.dataset):
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

        # generate summary pool
        beam_mult_output = model.generate(
            original_doc_token_ids[None, :],
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_seqs,
            max_length=150,
            early_stopping=args.early_stopping,
            # return_dict_in_generate=True,
            # output_scores=True,
        )

        beam_gen_sequences.append(beam_mult_output.cpu())

    save_to_cache_dir(
        beam_gen_sequences, 
        f"beam_gen_sequences_{args.num_beams}",
        cache_dir)

    exit()

    gen_count = 0
    target_count = args.num_return_seqs
    max_length = 100
    sequences_dump = []


    while gen_count < target_count:
        sample_multi_output = model.generate(
            original_doc_token_ids[None, :],
            do_sample=True, 
            max_length=100, 
            top_p=0.92, 
            # top_k=30,
            num_return_sequences=args.num_return_seqs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        unique_output = torch.unique(sample_multi_output.sequences, dim=0)
        print(unique_output.shape)
        sequences_dump.append(unique_output)
        gen_count += unique_output.shape[0]
        print("gen_count", gen_count)


    for dump in sequences_dump:
        dump_seq_num, dump_max_length = dump.size()
        dump_padding = torch.ones(dump_seq_num, max_length-dump_max_length).to(device)
        dump_padded = torch.cat((dump, dump_padding), dim=1).long()
    breakpoint()

    # actual text summaries
    gen_sequences_text = decode_mult_seqs(sequences)
    print(gen_sequences_text)
    
    # remove sos token to make labels
    gen_sequences = sequences[:, 1:]
    gen_labels = gen_sequences.masked_fill(
        gen_sequences == tokenizer.pad_token_id, -100
    )
    # actual text summaries
    gen_sequences_text = decode_mult_seqs(sequences)
    print(gen_sequences_text)

    print("check if unique")
    breakpoint()
    print(torch.unique(sequences, dim=0).shape == sequences.shape)
    
    # save metadata
    # gen_sequences_scores = beam_mult_output.sequences_scores

    # feed the documents to the model to get log probs
    with torch.no_grad():
        model_mult_output = model(
            input_ids=original_doc_token_ids.repeat(args.num_return_seqs, 1),
            labels=gen_labels,
        )

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    # losses of sequences, shape: [num_return_seqs, max_seq_len]
    # loss is negative log probability
    seq_losses = criterion(model_mult_output.logits.permute(0, 2, 1), gen_labels)
    seq_losses_masked = seq_losses.masked_fill(seq_losses==0., torch.nan)  # mask 0 with nan to ignore padding
    
    # log probabilities of sequences, shape: [num_return_seqs]
    seq_logprobs = -seq_losses_masked.nansum(1)
