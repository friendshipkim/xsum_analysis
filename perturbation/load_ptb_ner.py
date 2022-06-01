"""
Load ptb ner doc list
rename ood -> ptb
save it
"""

from utils import load_from_cache_dir, save_to_cache_dir

# hyperparameters
ner_cache_dir = "/home/wk247/workspace/xsum_analysis/cache/ptb_docs/ner"

# ptb docs list
def load_ptb_docs_list():
    ptb_docs_list = load_from_cache_dir(f"ptb_docs_list_trf", ner_cache_dir)
    return ptb_docs_list


def rename_ptb_docs_list(old_ptb_docs_list):
    new_ptb_docs_list = []
    for old_ptb_dict in old_ptb_docs_list:
        if old_ptb_dict == None:
            new_ptb_docs_list.append({})
            continue

        new_ptb_dict = {
            "original_id": old_ptb_dict["original_id"],
            "ptb_doc": old_ptb_dict["ood_doc"],
            "ptb_true_summary": old_ptb_dict["ood_true_summary"],
            "metadata": old_ptb_dict["metadata"],
        }

        new_ptb_docs_list.append(new_ptb_dict)

    assert len(old_ptb_docs_list) == len(new_ptb_docs_list)
    return new_ptb_docs_list


def save_ptb_docs_list(new_ptb_docs_list):
    save_to_cache_dir(new_ptb_docs_list, f"ptb_docs_list_trf", ner_cache_dir)


if __name__ == "__main__":
    old_ptb_docs_list = load_ptb_docs_list()
    breakpoint()
    new_ptb_docs_list = rename_ptb_docs_list(old_ptb_docs_list)
    breakpoint()
    save_ptb_docs_list(new_ptb_docs_list)
