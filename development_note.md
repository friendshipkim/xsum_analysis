# Development note

## 1. Create perturbed documents
1. insert
   * currently available options
     * insert_num_options = [1, 2]
     * insert_1_options = ["top1", "random"]
     * insert_2_options = ["topbottom", "random"]
   * how to cache
        ``` bash
        python perturbation/insert_off_topic.py --insert_num 1 --insert_position top1
        ```
2. named entity replacement
   * currently available options
     * insert_num_options = [1, 2]
     * insert_1_options = ["top1", "random"]
     * insert_2_options = ["topbottom", "random"]
   * how to cache
        ``` bash
        python perturbation/insert_off_topic.py --insert_num 1 --insert_position top1
        ```

### Data structure of cached perturbed documents
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

## 2. Generate summaries
```bash
bash ./scripts/cache_summary.sh -m facebook/bart-large-xsum -g topk -n 40 -k 100
```
* **true summary**
  * example
  ``` bash
  python generate_summary.py --gen_method true --num_return_seqs 1
  ```
* **greedy decoding** by calling `greedy_search()` if `num_beams=1` and `do_sample=False`.
  * TODO
* **multinomial sampling** by calling `sample()` if `num_beams=1` and `do_sample=True`.
  * top-k sampling
  ```bash
  python generate_summary.py --num_return_seqs 40 --gen_method topk --k 20 
  ```
  * top-p sampling
  ```bash
  python generate_summary.py --num_return_seqs 30 --gen_method topp --p 0.8 
  ```
* **beam-search decoding** by calling `beam_search()` if `num_beams>1` and `do_sample=False`.
  * example
  ``` bash
  python generate_summary.py --gen_method beam --num_return_seqs 30 --num_beams 30 
  ```
* **beam-search multinomial sampling** by calling `beam_sample()` if `num_beams>1` and `do_sample=True`.
  * TODO
* **diverse beam-search decoding** by calling `group_beam_search()`, if `num_beams>1` and `num_beam_groups>1`.
  * TODO
* **constrained beam-search decoding** by calling `constrained_beam_search()`, if `constraints!=None` or `force_words_ids!=None`.
  * TODO

## 3. Calculate log probabilities
```bash
# pass the arguments
bash ./scripts/cache_log_probs.sh -m facebook/bart-large-xsum -g topk -n 40 -k 100
```
or
```bash
# manually specify arguments in the file
bash ./scripts/cache_log_probs_sep.sh
```
1. Original documents
  ```bash
  python calculate_log_probs_original.py --gen_method topk --num_return_seqs 40 --k 100
  ```
  * change arguments for different generation methods and number of sequences

2. Perturbed documents
  * change arguments for different generation methods and number of sequences
  * Insert
  ```bash
  python calculate_log_probs_ptb.py --gen_method beam --num_return_seqs 30 --num_beams 30 --ptb_method insert --insert_num 1 --insert_position top1
  ```
  * NER
  ```bash
  python calculate_log_probs_ptb.py --gen_method beam --num_return_seqs 30 --num_beams 30 --ptb_method ner
  ```
  * change arguments for different perturbation methods