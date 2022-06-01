"""
# TODO: now only supports trf
randomly replace the named entity of test documents
"""
from os.path import join
from tqdm.notebook import tqdm

import datasets
from xsum_dataset import XsumDataset

import spacy
import config as cfg
from utils import load_from_cache_dir, save_to_cache_dir

import random
random.seed(cfg.seed)

# hyperparameters
ptb_docs_save_dir = join(cfg.ptb_docs_dir, "ner")
ner_cache_dir = join(cfg.ner_dir, cfg.ner_tagger)

# load dataset
xsum_data_raw = datasets.load_dataset("xsum")

# train/test data
xsum_val_data = XsumDataset(xsum_data_raw["validation"])
xsum_test_data = XsumDataset(xsum_data_raw["test"])

test_dataset = xsum_test_data.dataset
val_dataset = xsum_val_data.dataset

# load ner lists and pool
test_doc_ents_list = load_from_cache_dir(
    "test_doc_ents_list_no_dup", ner_cache_dir
)  # no duplicate
test_sum_ents_list = load_from_cache_dir("test_sum_ents_list_no_dup", ner_cache_dir)
val_test_ent_pool_dict = load_from_cache_dir("val_test_ent_pool_dict", ner_cache_dir)

# filter labels
NER = spacy.load(f"en_core_web_{cfg.ner_tagger}")
all_labels = list(NER.get_pipe("ner").labels)
filter_labels = cfg.filter_labels


# extract only necessary dicts
ent_pool_dict = val_test_ent_pool_dict
for label in all_labels:
    if label not in filter_labels:
        del ent_pool_dict[label]
assert len(ent_pool_dict) == len(filter_labels)

# reduce entity pool size
reduced_ent_pool_dict = {}
for label in ent_pool_dict.keys():
    ent_pool = ent_pool_dict[label]
    reduced_ent_pool = {
        k: v
        for (k, v) in list(ent_pool.items())[
            : int(len(ent_pool) * cfg.pool_size_reduction_ratio)
        ]
    }
    reduced_ent_pool_dict[label] = reduced_ent_pool


def if_improper_replacement(
    chosen_ent, chosen_label, replace_ent, document, true_summary
):
    # sample again if one of the below is true
    # 1. if label is person, replacement must have the same # of words
    # 2. if replace entity is in the true document
    # 3. if replace entity is in the true summary
    # 4. if replace entity is a subset of chosen entity
    # 5. if chosen entity is a subset of replace entity

    violations = [
        chosen_label == "PERSON"
        and len(replace_ent.split()) != len(chosen_ent.split()),
        replace_ent in document,
        replace_ent in true_summary,
        replace_ent in chosen_ent,
        chosen_ent in replace_ent,
    ]

    # if violations are all false -> return false
    # if one of them are true -> return true
    return any(violations)


# select entity and replace it
ptb_list = []
for data_idx, data in enumerate(tqdm(xsum_test_data.dataset)):
    print(f"\n============ data idx: {data_idx} ============")
    original_id = data["id"]
    original_doc = data["document"]
    true_summary = data["true_summary"]

    # ner
    # doc_ner = test_doc_ner_list[data_idx]
    # sum_ner = test_sum_ner_list[data_idx]

    # entites
    doc_ents = test_doc_ents_list[data_idx]
    sum_ents = test_sum_ents_list[data_idx]

    # sort and filter
    doc_ents_filtered_sorted = [
        ((ent, label), count)
        for ((ent, label), count) in doc_ents.most_common()
        if label in filter_labels
    ]
    sum_ents_filtered_sorted = [
        ((ent, label), count)
        for ((ent, label), count) in sum_ents.most_common()
        if label in filter_labels
    ]

    # if there is an overlap
    overlap_flag = False

    chosen_ent = None
    chosen_label = None

    # from document side
    # print(f"* summary: {true_summary}")
    # print(f"* document ents: {doc_ents_filtered_sorted} \n")
    for (ent, label), count in doc_ents_filtered_sorted:
        if ent in true_summary:  # overlap exists
            # print("** overlap from document")
            # print(f"ent: {ent}, label: {label}, count_doc: {count}, count_sum: {true_summary.count(ent)}")
            overlap_flag = True
            chosen_ent, chosen_label = ent, label
            break

    # if no overlap from document side, try from summary
    if overlap_flag == False:
        if len(sum_ents) == 0:  # no entity in summary -> pass
            pass
        else:
            # print(f"* summary ents: {sum_ents_filtered_sorted} \n")
            for (ent, label), count in sum_ents_filtered_sorted:
                if ent in original_doc:
                    # print("** overlap from summary")
                    # print(f"ent: {ent}, label: {label}, count_sum: {count}, count_doc: {original_doc.count(ent)}")
                    overlap_flag = True
                    chosen_ent, chosen_label = ent, label
                    break

    # check the chosen entity
    if overlap_flag == True:
        print(f"* summary: {true_summary}")
        print(f"* chosen_ent: {chosen_ent}, label: {chosen_label}")

        # choose one
        ent_pool = list(reduced_ent_pool_dict[chosen_label].keys())
        replace_ent = random.choice(ent_pool)
        print(f"* replace_ent: {replace_ent}")

        while if_improper_replacement(
            chosen_ent, chosen_label, replace_ent, original_doc, true_summary
        ):
            print("pick new replacement")
            replace_ent = random.choice(ent_pool)
            print(f"* replace_ent: {replace_ent}")

        metadata_dict = {
            "chosen_ent": chosen_ent,
            "replace_ent": replace_ent,
            "label": chosen_label,
        }

        ptb_doc = original_doc.replace(chosen_ent, replace_ent)
        ptb_summary = true_summary.replace(chosen_ent, replace_ent)
        ptb_list.append(
            {
                "original_id": original_id,
                "ptb_doc": ptb_doc,
                "ptb_true_summary": ptb_summary,
                "metadata": metadata_dict,
            }
        )
    else:
        print("****** NO OVERLAP ******")
        # if no overlap, append empty dictionary
        ptb_list.append({})

# save
# save_to_cache_dir(ptb_list, f"ptb_docs_list_{cfg.ner_tagger}", ptb_docs_save_dir)
