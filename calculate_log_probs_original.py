"""
Script to run inference (inputs: original document, labels: generated summaries) 
and save log probabilities of the generated summaries
This can be merged if log probs can be calculated while generating sequences
"""

import argparse
from os.path import join
from tqdm.contrib import tzip

import torch
import datasets
from xsum_dataset import XsumDataset

import config as cfg
from generate_xsum_summary import load_summarization_model_and_tokenizer
from utils import (
    calculate_log_probs,
    load_from_cache_dir,
    save_to_cache_dir,
    make_gen_seqs_filename,
    make_log_probs_filename,
)


def log_arguments(args, logger):
    logger.info(">> Calculating log probabilities of original documents")
    logger.info(">> Summary generation parameters:")
    logger.info(f"gen_method: {args.gen_method}")
    logger.info(f"num_return_seqs: {args.num_return_seqs}")

    if args.gen_method == "true":
        pass
    elif args.gen_method == "beam":
        logger.info(f"num_beams: {args.num_beams}")
    elif args.gen_method == "topk":
        logger.info(f"k: {args.k}")
    elif args.gen_method == "topp":
        logger.info(f"p: {args.p}")
    else:
        assert False, f"{args.gen_method} is invalid"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run inference on the generated summaries and save log probabilities of them"
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
        "--num_return_seqs",
        type=int,
        required=True,
        help="Number of generated summaries, if gen_method=='true', should be 1",
    )

    # arguments for generated sequences
    # beam search
    parser.add_argument(
        "--num_beams", type=int, required=False, help="Beam size",
    )

    # topk
    parser.add_argument(
        "--k", type=int, required=False, help="K for top-k sampling ",
    )

    # topp
    parser.add_argument(
        "--p", type=float, required=False, help="Probability for top-p sampling ",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # logging
    logger = cfg.log_probs_logger
    log_arguments(args, logger)

    # load model and tokenizer
    model, tokenizer = load_summarization_model_and_tokenizer(args.model_name)

    # load test dataset
    xsum_data_raw = datasets.load_dataset("xsum")
    xsum_test_dataset = XsumDataset(xsum_data_raw["test"]).dataset

    # load sequences list
    gen_seqs_filename = make_gen_seqs_filename(args)
    gen_seqs_list = load_from_cache_dir(gen_seqs_filename, cfg.gen_seqs_dir, logger)

    # ======= calaulate log probs of each sequences
    original_log_probs_list = []
    for i, (data, gen_seqs) in enumerate(tzip(xsum_test_dataset, gen_seqs_list)):
        # log progress
        if i % cfg.log_interval == 0:
            logger.info(f"{i} samples processed")

        original_id = data["id"]
        original_doc = data["document"]
        true_summary = data["true_summary"]

        # tokenize original document
        inputs = tokenizer(
            original_doc, truncation=True, return_tensors="pt", padding=True,
        )
        original_doc_token_ids = inputs.input_ids.to(cfg.device)

        # process generated sequences to use them as labels
        gen_seqs_clean = gen_seqs[:, 1:]  # remove <sos> token to make labels
        gen_labels = gen_seqs_clean.masked_fill(
            gen_seqs_clean == tokenizer.pad_token_id, cfg.mask_idx
        ).to(cfg.device)  # pad with -100

        # feed the document to the model to get log probs
        with torch.no_grad():
            original_model_output = model(
                input_ids=original_doc_token_ids.repeat(gen_labels.size(0), 1),
                labels=gen_labels,
            )
        original_log_probs = calculate_log_probs(
            original_model_output.logits, gen_labels
        ).cpu()

        # append to list
        # num_return_seqs may vary
        # assert args.num_return_seqs == original_log_probs.size(0)
        original_log_probs_list.append(original_log_probs)

    assert len(original_log_probs_list) == len(xsum_test_dataset)

    # ======== save it to log probs dir
    # directory differs from generation methods and the number of summaries
    save_dir = gen_seqs_filename.replace("gen_seqs_", "")
    save_to_cache_dir(
        original_log_probs_list,
        make_log_probs_filename(is_original=True, args=args),
        join(cfg.log_probs_dir, save_dir),
        logger,
    )
