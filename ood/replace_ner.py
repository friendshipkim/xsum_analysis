"""
# TODO: refer to ../notebooks/ner_xsum_analysis.ipynb
randomly replace the named entity of test documents
"""

import datasets
from xsum_dataset import XsumDataset
import random
from tqdm.notebook import tqdm
from os.path import exists, join
import spacy

from ner_utils import load_from_cache_dir, save_to_cache_dir

random.seed(0)

# hyperparameters
cache_dir = "/home/wk247/workspace/xsum_analysis/cache/ood_ner"
ner_cache_dir = "/home/wk247/workspace/xsum_analysis/cache_trf"
pool_size_reduction_ratio = 0.1

# load dataset
xsum_data_raw = datasets.load_dataset("xsum")

# train/test data
xsum_val_data = XsumDataset(xsum_data_raw["validation"])
xsum_test_data = XsumDataset(xsum_data_raw["test"])

test_dataset = xsum_test_data.dataset
val_dataset = xsum_val_data.dataset

# load ner lists and pool
test_doc_ents_list = load_from_cache_dir("test_doc_ents_list_no_dup", ner_cache_dir)  # no duplicate
test_sum_ents_list = load_from_cache_dir("test_sum_ents_list_no_dup", ner_cache_dir)
val_test_ent_pool_dict = load_from_cache_dir("val_test_ent_pool_dict", ner_cache_dir)

# filter labels
ALL_LABELS = list(NER.get_pipe('ner').labels)
FILTER_LABELS = ["PERSON", "FAC", "GPE", "NORP", "LOC", "EVENT", "LANGUAGE", "LAW", "ORG"]


# extract only necessary dicts
ent_pool_dict = val_test_ent_pool_dict
for label in ALL_LABELS:
    if label not in FILTER_LABELS:
        del ent_pool_dict[label]
assert(len(ent_pool_dict) == len(FILTER_LABELS))

# reduce entity pool size
reduced_ent_pool_dict = {}
for label in ent_pool_dict.keys():
    ent_pool = ent_pool_dict[label]
    reduced_ent_pool = {k:v for (k,v) in list(ent_pool.items())[:int(len(ent_pool) * pool_size_reduction_ratio)]}
    reduced_ent_pool_dict[label] = reduced_ent_pool

breakpoint()

def if_improper_replacement(chosen_ent, chosen_label, replace_ent, document, true_summary):
    # sample again if one of the below is true
    # 1. if label is person, replacement must have same # of words
    # 2. if replace entity is in the true document
    # 3. if replace entity is in the true summary
    # 4. if replace entity is a subset of chosen entity
    # 5. if chosen entity is a subset of replace entity
    
    violations = [chosen_label == "PERSON" and len(replace_ent.split()) != len(chosen_ent.split()),
                   replace_ent in document,
                   replace_ent in true_summary,
                   replace_ent in chosen_ent,
                   chosen_ent in replace_ent]
    
    # if violations are all false -> return false
    # if one of them are true -> return true
    return any(violations)

replacement_info = []
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
    doc_ents_filtered_sorted = [((ent, label), count) for ((ent, label), count) in doc_ents.most_common() 
                                if label in FILTER_LABELS]
    sum_ents_filtered_sorted = [((ent, label), count) for ((ent, label), count) in sum_ents.most_common() 
                                if label in FILTER_LABELS]
    
    # if there is an overlap
    overlap_flag = False
    
    
    chosen_ent = None
    chosen_label = None
    
    # from document side
#     print(f"* summary: {true_summary}")
#     print(f"* document ents: {doc_ents_filtered_sorted} \n")
    for (ent, label), count in doc_ents_filtered_sorted:
        if ent in true_summary:  # overlap exists
#             print("** overlap from document")
#             print(f"ent: {ent}, label: {label}, count_doc: {count}, count_sum: {true_summary.count(ent)}")
            overlap_flag = True
            chosen_ent, chosen_label = ent, label
            break
    
    # if no overlap from document side, try from summary
    if overlap_flag == False:
        if len(sum_ents) == 0: # no entity in summary -> pass
            pass
        else:
#             print(f"* summary ents: {sum_ents_filtered_sorted} \n")
            for (ent, label), count in sum_ents_filtered_sorted:
                if ent in original_doc:
#                     print("** overlap from summary")
#                     print(f"ent: {ent}, label: {label}, count_sum: {count}, count_doc: {original_doc.count(ent)}")
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
        
        while if_improper_replacement(chosen_ent, chosen_label, replace_ent, original_doc, true_summary):
            print("pick new replacement")
            replace_ent = random.choice(ent_pool)
            print(f"* replace_ent: {replace_ent}")
            
        
        replacement_info.append({"chosen_ent": chosen_ent,
                                 "replace_ent": replace_ent,
                                 "label": chosen_label,})
        
    else:
        print("****** NO OVERLAP ******")
        replacement_info.append({"chosen_ent": None,
                                 "replace_ent": None,
                                 "label": None,})


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
    