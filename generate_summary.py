import argparse
from tqdm import tqdm

import datasets
from xsum_dataset import XsumDataset

import config as cfg
from generate_xsum_summary import load_summarization_model_and_tokenizer
from utils import save_to_cache_dir

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ========== default parameters ==========
# model_name = "facebook/bart-large-xsum"
# gen_seqs_dir = "/home/wk247/workspace/xsum_analysis/cache/gen_seqs"
# max_summary_length = 150

# summary_generation_methods = ["true", "beam", "topp", "topk"]
# seed = 0
# # ========================================


# not used for now
# ========== generate methods
def generate_seqs_beam():
    beam_mult_output = model.generate(
        input_ids=original_doc_token_ids,
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        max_length=args.max_length,
        early_stopping=args.early_stopping,
        # return_dict_in_generate=True,
        # output_scores=True,
    )
    return beam_mult_output


def generate_seqs_topk():
    topk_multi_output = model.generate(
        input_ids=original_doc_token_ids,
        do_sample=True,
        max_length=max_length,
        top_k=args.k,
        num_return_sequences=args.num_return_seqs,
    )
    return topk_multi_output


def generate_seqs_topp():
    topp_multi_output = model.generate(
        input_ids=original_doc_token_ids,
        do_sample=True,
        max_length=max_length,
        top_k=0,
        top_p=args.probs,
        num_return_sequences=args.num_return_seqs,
    )
    return topp_multi_output


# =====================


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

    # beam search method
    parser.add_argument(
        "--num_beams",  # now num_return_seqs = num_beams
        type=int,
        required=False,
        help="Beam size",
    )

    parser.add_argument(
        "--early_stopping",
        type=bool,
        required=False,
        default=True,
        help="Whether to ealy stop the beam search (default: True))",
    )

    # topp
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=cfg.seed,
        help=f"Torch random seed (default: {cfg.seed})",
    )

    parser.add_argument(
        "--prob", type=float, required=False, help="Probability for top-p sampling ",
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

    elif args.gen_method == "topp" and (args.seed is None or args.prob is None):
        parser.error("'topp' method requires --seed and --prob")

    return args


if __name__ == "__main__":
    args = parse_args()

    # set random seed
    if args.seed:
        torch.random.seed = args.seed

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
        )

        exit()

    # TODO
    # model generate arguments
    gen_args = {"num_return_sequences": args.num_return_seqs}

    if args.gen_method == "beam":
        # gen_seqs = generate_seqs_beam()
        gen_args["do_sample"] = False
        gen_args["num_beams"] = args.num_beams
        gen_args["early_stopping"] = args.early_stopping
    elif args.gen_method == "topk":
        # gen_seqs = generate_seqs_topk()
        gen_args["do_sample"] = True
        gen_args["top_k"] = args.k
    elif args.gen_method == "topp":
        # gen_seqs = generate_seqs_topp()
        gen_args["do_sample"] = True
        gen_args["top_p"] = args.probs

    # ======= generate summaries with beam search
    gen_seqs_list = []
    for i, data in enumerate(xsum_test_dataset):
        # print progress
        if i % 1000 == 0:
            print(f"{i} samples processed")

        original_doc = data["document"]
        # true_summary = data["true_summary"]

        # tokenize original document
        inputs = tokenizer(
            original_doc,
            # max_length=1024,  # default is 1024 for 'facebook/bart-large-xsum'
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        original_doc_token_ids = inputs.input_ids.to(device)

        # generate summaries
        gen_args["input_ids"] = original_doc_token_ids
        gen_seqs = model.generate(**gen_args)

        assert gen_seqs.size(0) == args.num_return_seqs
        gen_seqs_list.append(gen_seqs.cpu())

    assert len(gen_seqs_list) == len(xsum_test_dataset)

    # save it to cache dir
    # TODO: file naming rule
    save_to_cache_dir(
        gen_seqs_list,
        f"gen_seqs_{args.gen_method}_{args.num_return_seqs}",
        cfg.gen_seqs_dir,
    )

    exit()

    # TODO: handle duplicates
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
        dump_padding = torch.ones(dump_seq_num, max_length - dump_max_length).to(device)
        dump_padded = torch.cat((dump, dump_padding), dim=1).long()
    breakpoint()
