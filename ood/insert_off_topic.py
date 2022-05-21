"""
randomly pick n sentences from the validation documents
and insert them to test documents
"""

import datasets
from xsum_dataset import XsumDataset
import random
from tqdm.notebook import tqdm
from os.path import exists, join

from ner_utils import load_from_cache_dir, save_to_cache_dir

random.seed(0)

# hyperparameters
sample_sents_n = 1
insert_position = "top1"  # random, top_bottom, top2
max_insert_sent_n = 10
cache_dir = "/home/wk247/workspace/xsum_analysis/cache/ood_insert"

# load dataset
xsum_data_raw = datasets.load_dataset("xsum")

# train/test data
xsum_val_data = XsumDataset(xsum_data_raw["validation"])
xsum_test_data = XsumDataset(xsum_data_raw["test"])

test_dataset = xsum_test_data.dataset
val_dataset = xsum_val_data.dataset

# load or make new ood random metadata
if exists(join(cache_dir, "ood_random_metadata.pkl")):
    ood_random_metadata = load_from_cache_dir("ood_random_metadata", cache_dir)
else:
    ood_random_metadata = []

    ood_random_id_pool = [
        id for id in xsum_val_data.ids 
        if len(xsum_val_data.data_by_id[id]["document"].split("\n")) > max_insert_sent_n
    ]
    ood_random_ids = random.choices(ood_random_id_pool, k=len(xsum_test_data))
    
    for ood_id in ood_random_ids:
        ood_selected_data = xsum_val_data.data_by_id[ood_id]
        ood_doc = ood_selected_data["document"]
        
        # store shuffled sentence order
        select_order = list(range(len(ood_doc.split("\n"))))
        random.shuffle(select_order)
        
        ood_random_metadata.append({
            "ood_id": ood_id, 
            "random_select_order": select_order
        })
    save_to_cache_dir(ood_random_metadata, "ood_random_metadata", cache_dir)
assert len(ood_random_metadata) == len(xsum_test_data)



ood_list = []
for test_data, ood_random_dict in tqdm(zip(test_dataset, ood_random_metadata)):
    metadata_dict = {}
    
    # get original docs
    original_id = test_data["id"]
    original_doc = test_data["document"]
    
    # get val data to extract ood sentence
    ood_id = ood_random_dict["ood_id"]
    ood_random_select_order = ood_random_dict["random_select_order"]

    ood_selected_data = xsum_val_data.data_by_id[ood_id]
    ood_doc = ood_selected_data["document"]

    # extract n sentences from the ood_doc
    ood_sents = ood_doc.split("\n")
    ood_insert_indices = ood_random_select_order[:sample_sents_n]
    ood_insert_sents = [ood_sents[i] for i in ood_insert_indices]
    
    # original doc sentences
    original_sents = original_doc.split("\n")

    # indices to insert
    if insert_position == "random":
        print("insert sentences into random positions")
        insert_indices = random.choices(range(len(original_sents)), k=sample_sents_n)
    elif insert_position == "top1":
        assert sample_sents_n == 1
        insert_indices = [0]
    elif insert_position == "topbottom":
        assert sample_sents_n == 2
        insert_indices = [0, -1]
    elif insert_position == "top2":
        assert sample_sents_n == 2
        insert_indices = [0, 1]
    else:
        print(f"{insert_position} is not supported")
        exit()
    
    # save metadata
    metadata_dict["original_doc_len"] = len(original_sents)
    metadata_dict["ood_id"] = ood_id
    metadata_dict["ood_sents_indices"] = ood_insert_indices
    metadata_dict["insert_indices"] = insert_indices

    for insert_idx, ood_sent in zip(insert_indices, ood_insert_sents):
        original_sents.insert(insert_idx, ood_sent)

    if len(original_sents) != metadata_dict["original_doc_len"] + sample_sents_n:
        print(len(original_sents), metadata_dict["original_doc_len"] + sample_sents_n)
        breakpoint()
    
    ood_doc = u"\n".join(original_sents)

    ood_list.append({
        "original_id": original_id,
        "ood_doc": ood_doc, 
        "metadata": metadata_dict
    })

save_to_cache_dir(ood_list, f"ood_list_{sample_sents_n}_{insert_position}", cache_dir)
    