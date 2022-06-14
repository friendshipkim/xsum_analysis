"""
Script to run inference (inputs: perturbed document, labels: generated summaries) 
and save log probabilities of the generated summaries
This can be merged if log probs can be calculated while generating sequences
"""

import argparse
from os.path import join
from tqdm.contrib import tzip
from typing import List

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
    make_ptb_docs_filename,
    make_log_probs_filename,
)

# decode multiple sequences
def decode_mult_seqs(
    seq_tokens: torch.LongTensor, skip_special_tokens: bool = True
) -> List[str]:
    return tokenizer.batch_decode(
        seq_tokens,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def log_arguments(args, logger):
    logger.info(">> Calculating log probabilities of perturbed documents")
    logger.info(">> Summary generation parameters:")
    logger.info(f"gen_method: {args.gen_method}")
    logger.info(f"num_return_seqs: {args.num_return_seqs}")

    # gen seqs
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

    # ptb
    logger.info(">> Document perturbation parameters:")
    logger.info(f"ptb_method: {args.ptb_method}")
    if args.ptb_method == "insert":
        logger.info(f"insert_num: {args.insert_num}")
        logger.info(f"insert_position: {args.insert_position}")
    elif args.ptb_method == "ner":
        logger.info(f"ner_tagger: {args.ner_tagger}")
    else:
        assert False, f"{args.ptb_method} is invalid"


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

    # arguments for perturbation
    parser.add_argument(
        "--ptb_method",
        type=str,
        required=True,
        choices=cfg.document_ptb_methods,
        help=f"Method used to perturb input documents (Choices: [{cfg.document_ptb_methods}])",
    )

    # arguments for insertion perturbation
    parser.add_argument(
        "--insert_num",
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

    # argument sanity check
    if args.ptb_method == "insert" and (
        args.insert_num is None or args.insert_position is None
    ):
        parser.error("'insert' method requires --insert_num and --insert_position")

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

    # load perturbed documents list
    ptb_docs_list = load_from_cache_dir(
        make_ptb_docs_filename(args), join(cfg.ptb_docs_dir, args.ptb_method), logger
    )

    assert len(gen_seqs_list) == len(ptb_docs_list)

    # ======= calaulate log probs of each sequences
    ptb_log_probs_list = []
    for i, (ptb_doc_dict, gen_seqs) in enumerate(tzip(ptb_docs_list, gen_seqs_list)):
        # log progress
        if i % cfg.log_interval == 0:
            logger.info(f"{i} samples processed")

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
                ptb_gen_seqs_text, truncation=True, return_tensors="pt", padding=True,
            )
            gen_seqs = ptb_inputs.input_ids

        else:
            ptb_doc = ptb_doc_dict["ptb_doc"]

        # tokenize ptb document
        inputs = tokenizer(ptb_doc, truncation=True, return_tensors="pt", padding=True,)
        ptb_doc_token_ids = inputs.input_ids.to(cfg.device)

        # process generated sequences to use them as labels
        gen_seqs_clean = gen_seqs[:, 1:]  # remove <sos> token to make labels
        gen_labels = gen_seqs_clean.masked_fill(
            gen_seqs_clean == tokenizer.pad_token_id, cfg.mask_idx
        ).to(
            cfg.device
        )  # pad with -100

        # feed the document to the model to get log probs
        with torch.no_grad():
            ptb_model_output = model(
                input_ids=ptb_doc_token_ids.repeat(gen_labels.size(0), 1),
                labels=gen_labels,
            )
        ptb_log_probs = calculate_log_probs(ptb_model_output.logits, gen_labels).cpu()

        # append to list
        # num_return_seqs may vary
        # assert args.num_return_seqs == ptb_log_probs.size(0)
        ptb_log_probs_list.append(ptb_log_probs)

    assert len(ptb_log_probs_list) == len(xsum_test_dataset)

    # ======== save it to log probs dir
    # directory differs from generation methods and the number of summaries
    save_dir = gen_seqs_filename.replace("gen_seqs_", "")
    save_to_cache_dir(
        ptb_log_probs_list,
        make_log_probs_filename(is_original=False, args=args),
        join(cfg.log_probs_dir, save_dir),
        logger,
    )
