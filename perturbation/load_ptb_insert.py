"""
Load ptb insert doc list and metadata
rename ood -> ptb
save it
"""

from utils import load_from_cache_dir, save_to_cache_dir

# hyperparameters
insert_n = 2
insert_position = "top2"
insert_cache_dir = "/home/wk247/workspace/xsum_analysis/cache/ptb_docs/insert"

# ptb docs list
def load_ptb_docs_list(insert_n, insert_position):
    ptb_docs_list = load_from_cache_dir(
        f"ptb_docs_list_{insert_n}_{insert_position}", insert_cache_dir
    )
    return ptb_docs_list


def rename_ptb_docs_list(old_ptb_docs_list):
    new_ptb_docs_list = []
    for old_ptb_dict in old_ptb_docs_list:
        old_metadata_dict = old_ptb_dict["metadata"]

        new_ptb_dict = {
            "original_id": old_ptb_dict["original_id"],
            "ptb_doc": old_ptb_dict["ood_doc"],
            "metadata": {
                "original_doc_len": old_metadata_dict["original_doc_len"],
                "ptb_id": old_metadata_dict["ood_id"],
                "ptb_sents_indices": old_metadata_dict["ood_sents_indices"],
                "insert_indices": old_metadata_dict["insert_indices"],
            },
        }

        new_ptb_docs_list.append(new_ptb_dict)

    assert len(old_ptb_docs_list) == len(new_ptb_docs_list)
    return new_ptb_docs_list


def save_ptb_docs_list(new_ptb_docs_list):
    save_to_cache_dir(
        new_ptb_docs_list,
        f"ptb_docs_list_{insert_n}_{insert_position}",
        insert_cache_dir,
    )


# ptb metadata
def load_insert_metadata():
    insert_metadata = load_from_cache_dir("ptb_random_metadata", insert_cache_dir)
    return insert_metadata


def rename_insert_metadata(old_insert_metadata):
    new_insert_metadata = []
    for old_dict in old_insert_metadata:
        new_insert_metadata.append(
            {
                "ptb_id": old_dict["ood_id"],
                "random_select_order": old_dict["random_select_order"],
            }
        )

    assert len(old_insert_metadata) == len(new_insert_metadata)
    return new_insert_metadata


def save_insert_metadata(new_insert_metadata):
    save_to_cache_dir(new_insert_metadata, f"ptb_random_metadata", insert_cache_dir)


if __name__ == "__main__":
    old_insert_metadata = load_insert_metadata()
    new_insert_metadata = rename_insert_metadata(old_insert_metadata)
    breakpoint()
    save_insert_metadata(new_insert_metadata)
