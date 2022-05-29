# Development note

## Cached data structure
* Data structure of ptb_random_metadata
``` python
{"ptb_id": str: xsum val bbcid used to sample off-topic sentences
 "random_select_order": List: order to insert off-topic sentences sequentially}
```

* Data structure of ptb_docs_list
1. insert
    ``` python
    {"original_id": str: original xsum test bbcid,
     "ptb_doc": str: perturbed document,
     "metadata": {
         "original_doc_len": int: original document length
         "ptb_id": str: xsum val bbcid used to sample off-topic sentences
         "ptb_sents_indices": List: sentence indices of the off-topic document
         "insert_indices": List: indices where off-topic sentences are inserted to the original document
     }}
    ```
2. named entity replacement
    ``` python
    {"original_id": str: original xsum test bbcid,
     "ptb_doc": str: perturbed document,
     "ptb_true_summary": str: true summary after replacing named entity,
     "metadata": {
         "original_doc_len": int: original document length
         "ptb_id": str: xsum val bbcid used to sample off-topic sentences
         "ptb_sents_indices": List: sentence indices of the off-topic document
         "insert_indices": List: indices where off-topic sentences are inserted to the original document
     }}
    ```

## Summary generation methods
* **greedy decoding** by calling `greedy_search()` if `num_beams=1` and `do_sample=False`.
* **multinomial sampling** by calling `sample()` if `num_beams=1` and `do_sample=True`.
* **beam-search decoding** by calling `beam_search()` if `num_beams>1` and `do_sample=False`.
  * example
  ``` bash
  python generate_summary.py --gen_method beam --num_return_seqs 40 --num_beams 40 
  ```
* **beam-search multinomial sampling** by calling `beam_sample()` if `num_beams>1` and `do_sample=True`.
* **diverse beam-search decoding** by calling `group_beam_search()`, if `num_beams>1` and `num_beam_groups>1`.
* **constrained beam-search decoding** by calling `constrained_beam_search()`, if `constraints!=None` or `force_words_ids!=None`.

