import pickle
import os
import torch
import logging
import argparse
import numpy as np
from torch import nn, Tensor
from typing import List, Type

import config as cfg

import bert_score
from rouge_score import rouge_scorer

# criterion
criterion = nn.CrossEntropyLoss(ignore_index=cfg.mask_idx, reduction="none")

# =========== log probs utils
def calculate_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Given logit tensor and labels, calculate log probs of each sequences
    Args:
        logits: logit tensor (shape: [num_seqs, max_seq_len, vocab_size])
        labels: label tensor (shape: [num_seqs, max_seq_len])
    Returns:
        seq_logprobs: torch.Tensor (shape: [num_seqs])
    """
    # losses of sequences, shape: [num_seqs, max_seq_len]
    # loss is negative log probability
    seq_losses = criterion(logits.permute(0, 2, 1), labels)
    seq_losses_masked = seq_losses.masked_fill(
        seq_losses == 0.0, torch.nan
    )  # mask 0 with nan to ignore padding

    # log probabilities of sequences, shape: [num_seqs]
    seq_logprobs = -seq_losses_masked.nansum(1)

    return seq_logprobs


# ========= caching utils
def save_to_cache_dir(
    var: object, file_name: str, cache_dir: str, logger: Type[logging.Logger]
) -> None:
    # if the directory doesn't exist, create it
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logger.info(f">> Directory '{cache_dir}' created")

    file_path = os.path.join(cache_dir, file_name + ".pkl")

    # check if the file already exists
    if os.path.exists(file_path):
        overwrite_flag = input(f"File '{file_path}' already exists, overwrite? [y/n]: ")
        if overwrite_flag == "y":
            with open(file_path, "wb") as f:
                pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f">> File '{file_path}' overwritten")
        elif overwrite_flag:
            logger.critical(f">> Do not overwrite, exit")
            exit()
    else:
        with open(file_path, "wb") as f:
            pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f">> File saved to '{file_path}'")


def load_from_cache_dir(
    file_name: str, cache_dir: str, logger: Type[logging.Logger]
) -> object:
    file_path = os.path.join(cache_dir, file_name + ".pkl")

    # check if file_path exists
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            var = pickle.load(f)
        logger.info(f">> File loaded from '{file_path}'")
        return var
    else:
        logger.critical(f">> File doesn't exist '{file_path}', exit")
        exit()


# ========= KL divergence utils
def calculate_KL(p_s: List, q_s: List, est_type: str = "basic") -> np.array:
    """
    given two lists of log prob tensors, return an array of KL-divergences
    Args:
        p_s: list of log prob tensors, (length: # of xsum test samples, log prob tensor shape: [num_seqs])
        q_s: list of log prob tensors, (length: # of xsum test samples, log prob tensor shape: [num_seqs])
        est_type: type of extimator (Default: basic)
    Returns:
        an array of KL divergences (length: # of xsum test samples)
    """
    assert len(p_s) == len(q_s)
    kl_list = []
    for p, q in zip(p_s, q_s):
        if len(q) == 0:  # for ner - skip this sample
            kl_list.append(np.nan)
            continue

        assert p.size(0) == q.size(0)
        num_y = p.size(0)

        if est_type == "basic":
            kl_list.append((torch.sum(p - q) / num_y).item())
        else:  # TODO: implement other kl estimators
            pass
    return np.array(kl_list)


# ========= filename utils
def make_gen_seqs_filename(args: Type[argparse.Namespace]) -> str:
    """
    given arguments, return the filename of generated sequences list
    Args:
        args (argparse.Namespace): argument namespace
    Returns:
        a filename
    """
    base_filename = f"gen_seqs_{args.gen_method}_{args.num_return_seqs}"
    if args.gen_method == "topk":
        filename = base_filename + f"_k{args.k}"
    elif args.gen_method == "topp":
        filename = base_filename + f"_p{args.p}"
    elif args.gen_method == "beam":
        filename = base_filename + f"_beam{args.num_beams}"
    elif args.gen_method == "true":
        filename = base_filename
    else:
        assert False, f"{args.gen_method} is invalid"
    return filename


def make_ptb_docs_filename(args: Type[argparse.Namespace]) -> str:
    """
    given arguments, return the filename of perturbed documents
    Args:
        args (argparse.Namespace): argument namespace
    Returns:
        a filename
    """
    if args.ptb_method == "ner":
        filename = f"ptb_docs_list_{args.ner_tagger}"
    elif args.ptb_method == "insert":
        filename = f"ptb_docs_list_{args.insert_num}_{args.insert_position}"
    return filename


def make_log_probs_filename(is_original: bool, args: Type[argparse.Namespace]) -> str:
    """
    given arguments, return the filename of perturbed documents
    Args:
        is_original (bool): if given document is original (true) or perturbed (false)
        args (argparse.Namespace): argument namespace
    Returns:
        a filename
    """
    # original doc
    if is_original:
        return "original_log_probs_list"

    # ptb docs
    if args.ptb_method == "ner":
        filename = f"ptb_log_probs_list_ner_{args.ner_tagger}"
    elif args.ptb_method == "insert":
        filename = f"ptb_log_probs_list_{args.ptb_method}_{args.insert_num}_{args.insert_position}"
    return filename


# ========= summary scoring utils
def entropy(p_dist: torch.Tensor) -> float:
    """"
    Calculates Shannon entropy for a probability distribution
    Args:
        p_dist: probability distribution (torch.Tensor)
    Returns:
        entropy (float)
    """
    # add epsilon because log(0) = nan
    p_dist = p_dist.view(-1) + 1e-12
    return -torch.mul(p_dist, p_dist.log()).sum(0).item()


def score_each(
    hyps, refs, metric="bertscore", model_type="microsoft/deberta-xlarge-mnli"
):
    """
    Compute the bert score or rough score for hypothesis and reference pairs.

    Args:
        hyps: a list of string, hypothesis
        refs: a list of string, references
        metric: metric to compute, bertsocre, rouge1, rouge2, rougeL, rougeLsum
        model_type: model to cacluate bertscore

    Returns:
        precisions, recalls, fmeasures
    """

    if metric == "bertscore":
        precisions, recalls, fmeasures = bert_score.score(
            hyps, refs, model_type=model_type, lang="en", verbose=True
        )
        return precisions.tolist(), recalls.tolist(), fmeasures.tolist()
    elif metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        scorer = rouge_scorer.RougeScorer([metric])
        precisions, recalls, fmeasures = [], [], []
        # for each of the hypothesis and reference documents pair
        for (h, r) in zip(hyps, refs):
            # computing the ROUGE
            score = scorer.score(h, r)
            # separating the measurements
            precision, recall, fmeasure = score[metric]
            precisions.append(precision)
            recalls.append(recall)
            fmeasures.append(fmeasure)
        return precisions, recalls, fmeasures
    else:
        raise ValueError("Metric is not implemented.")


def score(
    hyps, ref, metric="bertscore", model_type="microsoft/deberta-xlarge-mnli", lang="en"
):
    """
    Compute the bert score or rough score given a gold summary and a list of summaries generated by models.

    Args:
        hyps: a list of summaries generated by models
        ref: a gold summary
        metric: metric to compute, bertsocre, rouge1, rouge2, rougeL, rougeLsum
        model_type: model to cacluate bertscore

    Returns:
        precisions, recalls, fmeasures
    """
    refs = [ref] * len(hyps)
    return score_each(hyps, refs, metric, model_type)
