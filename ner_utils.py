import pickle
import os
from tqdm import tqdm
from collections import Counter


def save_to_cache_dir(var, file_name, cache_dir = "./cache"):
    file_path = os.path.join(cache_dir, file_name + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved to '{file_path}'")

def load_from_cache_dir(file_name, cache_dir = "./cache"):
    file_path = os.path.join(cache_dir, file_name + ".pkl")
    with open(file_path, 'rb') as f:
        var = pickle.load(f)
    print(f"'{file_path}' loaded")
    return var

def tag_dataset(tagger, xsum_dataset, return_ner_list=True):
    doc_ner_list = []
    sum_ner_list = []
    doc_ents_list = []
    sum_ents_list = []
    
    for data_idx, data in enumerate(xsum_dataset):
        document = data["document"]
        true_summary = data["true_summary"]

        # ner
        doc_ner = tagger(document.replace("\n", " "))  # removing newline changes the ner result
        sum_ner = tagger(true_summary)
        
        if return_ner_list: # train data is too large to store all ner objects
            doc_ner_list.append(doc_ner)
            sum_ner_list.append(sum_ner)

        # extract entities from document
        doc_ents = Counter([(ent.text, ent.label_) for ent in doc_ner.ents])

        # extract entities from summary
        sum_ents = Counter([(ent.text, ent.label_) for ent in sum_ner.ents])

        doc_ents_list.append(doc_ents)
        sum_ents_list.append(sum_ents)
    
    return doc_ner_list, sum_ner_list, doc_ents_list, sum_ents_list


def create_ent_pool_dict(doc_ents_list, labels):
    ent_pool = Counter()
    
    for doc_ents in tqdm(doc_ents_list):
        for e in doc_ents.elements():
            doc_ents[e] = 1  # set all counts as 1
        # add to entity pool
        ent_pool += doc_ents
    
    # entity pool dictionary
    ent_pool_dict = {l:{} for l in labels}
    
    # groupby entities by their names
    for (ent, label), count in ent_pool.items():
        ent_pool_dict[label][ent] = count
        
    # sort dicts by count
    for label in labels:
        ent_pool_dict[label] = dict(sorted(ent_pool_dict[label].items(), key=lambda item: item[1], reverse=True))
    
    # print entity dict info
    for label, li in ent_pool_dict.items():
        print(f"label: {label}, count: {len(li)}")
    
    return ent_pool_dict