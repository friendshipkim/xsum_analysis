import datasets
from xsum_dataset import XsumDataset
import spacy
import numpy as np

from utils import save_to_cache_dir
from ner_utils import tag_dataset

# load raw data
xsum_data_raw = datasets.load_dataset("xsum")

# create train XsumDataset
xsum_train_data = XsumDataset(xsum_data_raw["train"])

# NER
# NER = spacy.load("en_core_web_lg")
NER = spacy.load("en_core_web_trf")
cache_dir = "/home/wk247/workspace/xsum_analysis/cache/ner/trf/train"

# save interval
doc_n = len(xsum_train_data.dataset)
save_interval = 10000

start_idxs = np.arange(0, doc_n, save_interval)
end_idxs = np.append(np.arange(0, doc_n, save_interval)[1:], doc_n)
xsum_train_dataset = xsum_train_data.dataset

save_idx = 10  # split jobs

for start_idx, end_idx in zip(start_idxs[save_idx:], end_idxs[save_idx:]):
    _, _, train_doc_ents_list, train_sum_ents_list = tag_dataset(
        tagger=NER,
        xsum_dataset=xsum_train_dataset[start_idx: end_idx], 
        return_ner_list=False,
        )

    # save to cache directory
    save_to_cache_dir(
        train_doc_ents_list, 
        f"train_doc_ents_list_{start_idx}_{end_idx}",
        cache_dir)
    save_to_cache_dir(
        train_sum_ents_list, 
        f"train_sum_ents_list_{start_idx}_{end_idx}",
        cache_dir)