"""
Script to run inference (inputs: perturbed document, labels: generated summaries) 
and save log probabilities of the generated summaries
This can be merged if log probs can be calculated while generating sequences
"""

import argparse
from os.path import join
from tqdm import tqdm
from typing import List

import datasets
from xsum_dataset import XsumDataset

import config as cfg
from generate_xsum_summary import load_summarization_model_and_tokenizer
from utils import calculate_log_probs, load_from_cache_dir, save_to_cache_dir

import torch


# # ========== default parameters ==========
# model_name = "facebook/bart-large-xsum"

# ptb_docs_dir = "/home/wk247/workspace/xsum_analysis/cache/ptb_docs"
# gen_seqs_dir = "/home/wk247/workspace/xsum_analysis/cache/gen_seqs"
# log_probs_dir = "/home/wk247/workspace/xsum_analysis/cache/log_probs"

# summary_generation_methods = ["beam", "topp", "topk", "true"]
# document_ptb_methods = ["insert", "ner"]
# # ========================================

# decode multiple sequences
def decode_mult_seqs(
    seq_tokens: torch.LongTensor, skip_special_tokens: bool = True
) -> List[str]:
    return [
        tokenizer.decode(
            seq, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for seq in seq_tokens
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run inference on the generated summaries given perturbed documents and save log probabilities of them"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=cfg.model_name,
        help=f"Summarization model to test (Default: {cfg.model_name})",
    )

    parser.add_argument(
        "--gen_method",
        type=str,
        required=True,
        choices=cfg.summary_generation_methods,
        help=f"Method used to generate summaries (Choices: [{cfg.summary_generation_methods}])",
    )

    parser.add_argument(
        "--num_seqs", type=int, required=True, help="The number of generated summaries",
    )

    parser.add_argument(
        "--ptb_method",
        type=str,
        required=True,
        choices=cfg.document_ptb_methods,
        help=f"Method used to perturb input documents (Choices: [{cfg.document_ptb_methods}])",
    )

    # arguments for insertion perturbation
    parser.add_argument(
        "--num_insert",
        type=str,
        required=False,
        help="The number of inserted sentences",
    )

    parser.add_argument(
        "--insert_position",
        type=str,
        required=False,
        help="Position of inserted sentences",
    )

    # arguments for named entity replacement
    parser.add_argument(
        "--ner_tagger",
        type=str,
        required=False,
        default=cfg.ner_tagger,
        help=f"Type of NER tagger (Default: {cfg.ner_tagger})",
    )

    args = parser.parse_args()

    if args.ptb_method == "insert" and (
        args.num_insert is None or args.insert_position is None
    ):
        parser.error("'insert' method requires --num_insert and --insert_position")

    return args


if __name__ == "__main__":
    args = parse_args()

    # load model and tokenizer
    model, tokenizer = load_summarization_model_and_tokenizer(args.model_name)

    # load test dataset
    xsum_data_raw = datasets.load_dataset("xsum")
    xsum_test_dataset = XsumDataset(xsum_data_raw["test"]).dataset

    # load sequences list
    gen_seqs_list = load_from_cache_dir(
        f"gen_seqs_{args.gen_method}_{args.num_seqs}", cfg.gen_seqs_dir
    )

    # load perturbed documents list
    if args.ptb_method == "ner":
        ptb_docs_filename = f"ptb_docs_list_{args.ner_tagger}"
    elif args.ptb_method == "insert":
        ptb_docs_filename = f"ptb_docs_list_{args.num_insert}_{args.insert_position}"

    ptb_docs_list = load_from_cache_dir(
        ptb_docs_filename, join(cfg.ptb_docs_dir, args.ptb_method)
    )

    assert len(gen_seqs_list) == len(ptb_docs_list)

    # ======= calaulate log probs of each sequences
    ptb_log_probs_list = []
    for ptb_doc_dict, gen_seqs in tqdm(
        zip(ptb_docs_list, gen_seqs_list), total=len(gen_seqs_list)
    ):

        # if ptb_doc_dict is empty, append empty tensor and continue
        if len(ptb_doc_dict) == 0:
            ptb_log_probs_list.append(torch.Tensor([]))
            continue

        # ========= if ner, replace named entities in the generated sumamries
        if args.ptb_method == "ner":
            ptb_doc = ptb_doc_dict["ptb_doc"]
            ptb_true_summary = ptb_doc_dict["ptb_true_summary"]
            chosen_ent = ptb_doc_dict["metadata"]["chosen_ent"]
            replace_ent = ptb_doc_dict["metadata"]["replace_ent"]

            # decode generated summaries and replace named entities
            gen_seqs_text = decode_mult_seqs(gen_seqs)
            ptb_gen_seqs_text = [
                text.replace(chosen_ent, replace_ent) for text in gen_seqs_text
            ]

            # re-encode generated summaries
            ptb_inputs = tokenizer(
                ptb_gen_seqs_text,
                # max_length=1024,  # default is 1024 for 'facebook/bart-large-xsum'
                truncation=True,
                return_tensors="pt",
                padding=True,
            )
            gen_seqs = ptb_inputs.input_ids

        else:
            ptb_doc = ptb_doc_dict["ptb_doc"]

        # tokenize ptb document
        inputs = tokenizer(
            ptb_doc,
            # max_length=1024,  # default is 1024 for 'facebook/bart-large-xsum'
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        ptb_doc_token_ids = inputs.input_ids.to(cfg.device)

        # process generated sequences to use them as labels
        gen_seqs_clean = gen_seqs[:, 1:]  # remove <sos> token to make labels
        gen_labels = gen_seqs_clean.masked_fill(
            gen_seqs_clean == tokenizer.pad_token_id, cfg.mask_idx
        ).to(cfg.device)  # pad with -100

        # feed the document to the model to get log probs
        with torch.no_grad():
            ptb_model_output = model(
                input_ids=ptb_doc_token_ids.repeat(args.num_seqs, 1), labels=gen_labels,
            )
        ptb_log_probs = calculate_log_probs(ptb_model_output.logits, gen_labels).cpu()

        # append to list
        assert args.num_seqs == ptb_log_probs.size(0)
        ptb_log_probs_list.append(ptb_log_probs)

    assert len(ptb_log_probs_list) == len(xsum_test_dataset)

    # ======== save it to log probs dir
    # directory differs from generation methods and the number of summaries
    save_dir = join(cfg.log_probs_dir, f"{args.gen_method}_{args.num_seqs}")

    if args.ptb_method == "ner":
        ptb_log_probs_filename = (
            f"ptb_log_probs_list_ner_{args.ner_tagger}"
        )
    elif args.ptb_method == "insert":
        ptb_log_probs_filename = f"ptb_log_probs_list_{args.ptb_method}_{args.num_insert}_{args.insert_position}"

    save_to_cache_dir(ptb_log_probs_list, ptb_log_probs_filename, save_dir)
