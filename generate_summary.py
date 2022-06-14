import argparse
from tqdm import tqdm

import datasets
from xsum_dataset import XsumDataset

import config as cfg
from generate_xsum_summary import load_summarization_model_and_tokenizer
from utils import save_to_cache_dir, make_gen_seqs_filename

import torch


def log_arguments(args, logger):
    logger.info(">> Generating sequences")
    logger.info(f"gen_method: {args.gen_method}")
    logger.info(f"num_return_seqs: {args.num_return_seqs}")
    logger.info(f"max_length: {args.max_length}")

    if args.gen_method == "true":
        pass
    elif args.gen_method == "beam":
        logger.info(f"num_beams: {args.num_beams}")
        logger.info(f"early_stopping: {args.early_stopping}")
        logger.info(
            f"num_return_seqs_per_trial: {args.num_beams}"
        )  # same with num_beams
    elif args.gen_method == "topk":
        logger.info(f"seed: {args.seed}")
        logger.info(f"k: {args.k}")
        logger.info(f"num_return_seqs_per_trial: {args.num_return_seqs_per_trial}")
    elif args.gen_method == "topp":
        logger.info(f"seed: {args.seed}")
        logger.info(f"p: {args.p}")
        logger.info(f"num_return_seqs_per_trial: {args.num_return_seqs_per_trial}")
    else:
        assert False, f"{args.gen_method} is invalid"


def pad_sequences(seqs, pad_idx, max_length):
    seq_num, seq_len = seqs.size()
    padding = torch.ones(seq_num, max_length - seq_len).to(cfg.device) * pad_idx
    return torch.cat((seqs, padding), dim=1).long()


def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate summaries")

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=cfg.model_name,
        help=f"Summarization model to test (Default: {cfg.model_name})",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=cfg.max_summary_length,
        help=f"Maximum summary length (default: [{cfg.max_summary_length}]))",
    )

    parser.add_argument(
        "--gen_method",
        type=str,
        required=True,
        choices=cfg.summary_generation_methods,
        help=f"Method to generate summaries (Choices: [{cfg.summary_generation_methods}])",
    )

    parser.add_argument(
        "--num_return_seqs",
        type=int,
        required=True,
        help="The number of returned sequences (the size of summary pool), if gen_method='true', 1",
    )

    parser.add_argument(
        "--num_return_seqs_per_trial",
        type=int,
        required=False,
        default=cfg.num_return_seqs_per_trial,
        help=f"The number of returned sequences per trial (Default: beam - num_return_seq, sampling - {cfg.num_return_seqs_per_trial})",
    )

    # beam search
    parser.add_argument(
        "--num_beams", type=int, required=False, help="Beam size",
    )

    parser.add_argument(
        "--early_stopping",
        type=bool,
        required=False,
        default=True,
        help="Whether to ealy stop the beam search (default: True))",
    )

    # sampling
    parser.add_argument(
        "--max_trial",
        type=int,
        required=False,
        default=cfg.max_trial,
        help=f"The number of max trial for sampling (default: {cfg.max_trial}",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=cfg.seed,
        help=f"Torch random seed (default: {cfg.seed})",
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

    # arguments sanity check
    if args.gen_method == "beam":
        if args.num_beams is None or args.early_stopping is None:
            parser.error("'beam' method requires --num_beams and --early_stopping")
        if args.num_beams <= 1:
            parser.error("'beam' method requires beam size > 1")

    elif args.gen_method == "topk" and (args.seed is None or args.k is None):
        parser.error("'topk' method requires --seed and --k")

    elif args.gen_method == "topp" and (args.seed is None or args.p is None):
        parser.error("'topp' method requires --seed and --p")

    return args


if __name__ == "__main__":
    args = parse_args()

    # logging
    logger = cfg.gen_seqs_logger
    log_arguments(args, logger)

    # set random seed
    if args.seed:
        torch.random.set_seed(args.seed)

    # load model and tokenizer
    model, tokenizer = load_summarization_model_and_tokenizer(args.model_name)

    # load test dataset
    xsum_data_raw = datasets.load_dataset("xsum")
    xsum_test_dataset = XsumDataset(xsum_data_raw["test"]).dataset

    # if true summary, just save tokenized sequences and exit
    if args.gen_method == "true":
        true_summary_list = [data["true_summary"] for data in xsum_test_dataset]
        true_sequence_list = []
        for true_summary in true_summary_list:
            true_sequence = tokenizer(
                true_summary, truncation=True, return_tensors="pt", padding=True,
            ).input_ids
            true_sequence_list.append(true_sequence)

        # save it to cache dir
        save_to_cache_dir(
            true_sequence_list,
            f"gen_seqs_{args.gen_method}_{args.num_return_seqs}",
            cfg.gen_seqs_dir,
            logger,
        )

        exit()

    # sequence generate arguments
    gen_args = {}
    if args.gen_method == "beam":
        gen_args["num_return_sequences"] = args.num_return_seqs
        gen_args["do_sample"] = False
        gen_args["num_beams"] = args.num_beams
        gen_args["early_stopping"] = args.early_stopping
    elif args.gen_method == "topk":
        gen_args["num_return_sequences"] = args.num_return_seqs_per_trial
        gen_args["do_sample"] = True
        gen_args["top_k"] = args.k
        # gen_args["temperature"] = 0.5  # TBD
    elif args.gen_method == "topp":
        gen_args["num_return_sequences"] = args.num_return_seqs_per_trial
        gen_args["do_sample"] = True
        gen_args["top_p"] = args.p

    # ======= generate summaries with beam search
    gen_seqs_list = []
    for i, data in enumerate(tqdm(xsum_test_dataset)):
        # log progress
        if i % cfg.log_interval == 0:
            logger.info(f"{i} samples processed")

        original_doc = data["document"]
        # true_summary = data["true_summary"]

        # tokenize original document
        inputs = tokenizer(
            original_doc, truncation=True, return_tensors="pt", padding=True,
        )
        original_doc_token_ids = inputs.input_ids.to(cfg.device)

        # generate summaries
        gen_args["input_ids"] = original_doc_token_ids

        # repeat generation until we get num_return_seqs unique seqs
        final_gen_seqs = torch.empty((1, args.max_length))
        trial_count = 0
        while (
            final_gen_seqs.size(0) < args.num_return_seqs
            and trial_count <= args.max_trial
        ):
            # print("generate", final_gen_seqs.size(0))
            gen_seqs = model.generate(**gen_args)
            unique_gen_seqs = pad_sequences(
                torch.unique(gen_seqs, dim=0),
                pad_idx=tokenizer.pad_token_id,
                max_length=args.max_length,
            )
            trial_count += 1
            if trial_count == 1:
                final_gen_seqs = unique_gen_seqs
            else:
                final_gen_seqs = torch.cat((final_gen_seqs, unique_gen_seqs), dim=0)
                final_gen_seqs = torch.unique(final_gen_seqs, dim=0)
                # breakpoint()

        # save only num_return_seqs sequences
        # final_gen_seqs = final_gen_seqs[:args.num_return_seqs, :]
        # print("final", final_gen_seqs.size(0))
        gen_seqs_list.append(final_gen_seqs.cpu())

    assert len(gen_seqs_list) == len(xsum_test_dataset)

    # save it to cache dir
    save_to_cache_dir(
        gen_seqs_list, make_gen_seqs_filename(args), cfg.gen_seqs_dir, logger,
    )
