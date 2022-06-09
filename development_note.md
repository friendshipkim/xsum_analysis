# Development note

## Create perturbed documents
1. insert
   * currently available options
     * insert_num_options = [1, 2]
     * insert_1_options = ["top1", "random"]
     * insert_2_options = ["topbottom", "random"]
   * how to cache
        ``` bash
        python perturbation/insert_off_topic.py --num_insert 1 --insert_position top1
        ```
2. named entity replacement
   * currently available options
     * insert_num_options = [1, 2]
     * insert_1_options = ["top1", "random"]
     * insert_2_options = ["topbottom", "random"]
   * how to cache
        ``` bash
        python perturbation/insert_off_topic.py --num_insert 1 --insert_position top1
        ```

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
* **true summary**
  * example
  ``` bash
  python generate_summary.py --gen_method true --num_return_seqs 1
  ```
* **greedy decoding** by calling `greedy_search()` if `num_beams=1` and `do_sample=False`.
* **multinomial sampling** by calling `sample()` if `num_beams=1` and `do_sample=True`.
* **beam-search decoding** by calling `beam_search()` if `num_beams>1` and `do_sample=False`.
  * example
  ``` bash
  python generate_summary.py --gen_method beam --num_return_seqs 30 --num_beams 30 
  ```
* **beam-search multinomial sampling** by calling `beam_sample()` if `num_beams>1` and `do_sample=True`.
* **diverse beam-search decoding** by calling `group_beam_search()`, if `num_beams>1` and `num_beam_groups>1`.
* **constrained beam-search decoding** by calling `constrained_beam_search()`, if `constraints!=None` or `force_words_ids!=None`.

* top-k sampling
```bash
python generate_summary.py --num_return_seqs 20 --gen_method topk --k 20 
```

## Calculate log probabilities
1. Original documents
  ```bash
  bash scripts/cache_log_probs_original.sh
  ```
  * change arguments for different generation methods and number of sequences
2. Perturbed documents
   ```bash
  bash scripts/cache_log_probs_ptb.sh
  ```
  * change arguments for different generation methods and number of sequences