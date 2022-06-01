"""
randomly pick n sentences from the validation documents
and insert them into test documents
"""

import datasets
from xsum_dataset import XsumDataset

import argparse
import random
from tqdm.notebook import tqdm
from os.path import exists, join

import config as cfg
from utils import load_from_cache_dir, save_to_cache_dir

# ========== default parameters ==========
# insert_num_options = [1, 2]
# insert_1_options = ["top1", "random"]
# insert_2_options = ["topbottom", "random"]
# num_max_insert = 10
ptb_docs_save_dir = join(cfg.ptb_docs_dir, "insert")
# ========================================

# load dataset
xsum_data_raw = datasets.load_dataset("xsum")

# val/test data
xsum_val_data = XsumDataset(xsum_data_raw["validation"])
xsum_test_data = XsumDataset(xsum_data_raw["test"])

test_dataset = xsum_test_data.dataset
val_dataset = xsum_val_data.dataset


def load_random_metadata():
    # load or make new ptb random metadata
    if exists(join(ptb_docs_save_dir, "ptb_random_metadata.pkl")):
        ptb_random_metadata = load_from_cache_dir("ptb_random_metadata", ptb_docs_save_dir)
    else:
        ptb_random_metadata = []

        ptb_random_id_pool = [
            id
            for id in xsum_val_data.ids
            if len(xsum_val_data.data_by_id[id]["document"].split("\n"))
            > cfg.num_max_insert
        ]
        ptb_random_ids = random.choices(ptb_random_id_pool, k=len(xsum_test_data))

        for ptb_id in ptb_random_ids:
            ptb_selected_data = xsum_val_data.data_by_id[ptb_id]
            ptb_doc = ptb_selected_data["document"]

            # store shuffled sentence order
            select_order = list(range(len(ptb_doc.split("\n"))))
            random.shuffle(select_order)

            ptb_random_metadata.append(
                {"ptb_id": ptb_id, "random_select_order": select_order}
            )
        save_to_cache_dir(ptb_random_metadata, "ptb_random_metadata", ptb_docs_save_dir)
    assert len(ptb_random_metadata) == len(xsum_test_data)
    return ptb_random_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to insert off topic sentences to xsum test data"
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=cfg.seed,
        help=f"Random seed (default: {cfg.seed})",
    )

    # arguments for insertion perturbation
    parser.add_argument(
        "--num_insert",
        type=int,
        required=True,
        choices=cfg.insert_num_options,
        help=f"The number of inserted sentences (Choices: [{cfg.insert_num_options}])",
    )

    parser.add_argument(
        "--insert_position",
        type=str,
        required=True,
        help=f"Position of inserted sentences (Choices 1 - [{cfg.insert_1_options}], 2 - [{cfg.insert_2_options}])",
    )

    args = parser.parse_args()

    # arguments sanity check
    if args.num_insert == 1 and (args.insert_position not in cfg.insert_1_options):
        parser.error(
            f"if --num_insert == 1, --insert_position should be one of {cfg.insert_1_options}"
        )
    elif args.num_insert == 2 and (args.insert_position not in cfg.insert_2_options):
        parser.error(
            f"if --num_insert == 2, --insert_position should be one of {cfg.insert_2_options}"
        )

    return args


if __name__ == "__main__":
    args = parse_args()

    # set random seed
    random.seed(args.seed)

    # load random metadata to fix sentence insertion order
    ptb_random_metadata = load_random_metadata()

    # insert sentences
    ptb_list = []
    for test_data, ptb_random_dict in tqdm(zip(test_dataset, ptb_random_metadata)):
        metadata_dict = {}

        # get original docs
        original_id = test_data["id"]
        original_doc = test_data["document"]

        # get val data to extract ptb sentence
        ptb_id = ptb_random_dict["ptb_id"]
        ptb_random_select_order = ptb_random_dict["random_select_order"]

        ptb_selected_data = xsum_val_data.data_by_id[ptb_id]
        ptb_doc = ptb_selected_data["document"]

        # extract n sentences from the ptb_doc
        ptb_sents = ptb_doc.split("\n")
        ptb_insert_indices = ptb_random_select_order[: args.num_insert]
        ptb_insert_sents = [ptb_sents[i] for i in ptb_insert_indices]

        # original doc sentences
        original_sents = original_doc.split("\n")

        # indices to insert
        if args.insert_position == "random":
            print("insert sentences into random positions")
            insert_indices = random.choices(
                range(len(original_sents)), k=args.sample_sents_n
            )
        elif args.insert_position == "top1":
            assert args.num_insert == 1
            insert_indices = [0]
        elif args.insert_position == "topbottom":
            assert args.num_insert == 2
            insert_indices = [0, -1]
        elif args.insert_position == "top2":
            assert args.num_insert == 2
            insert_indices = [0, 1]
        else:
            print(f"{args.insert_position} is not supported")
            exit()

        # save metadata
        metadata_dict["original_doc_len"] = len(original_sents)
        metadata_dict["ptb_id"] = ptb_id
        metadata_dict["ptb_sents_indices"] = ptb_insert_indices
        metadata_dict["insert_indices"] = insert_indices

        # insert off-topic sentences
        for insert_idx, ptb_sent in zip(insert_indices, ptb_insert_sents):
            original_sents.insert(insert_idx, ptb_sent)

        # check error
        if len(original_sents) != metadata_dict["original_doc_len"] + args.num_insert:
            print(
                len(original_sents), metadata_dict["original_doc_len"] + args.num_insert
            )
            breakpoint()

        ptb_doc = "\n".join(original_sents)

        ptb_list.append(
            {"original_id": original_id, "ptb_doc": ptb_doc, "metadata": metadata_dict}
        )

    # save ptb_list to cache dir
    save_to_cache_dir(
        ptb_list, f"ptb_list_{args.num_insert}_{args.insert_position}", ptb_docs_save_dir
    )
