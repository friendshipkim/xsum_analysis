import pickle
import os
import torch
import numpy as np
from torch import nn, Tensor
from typing import List

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
def save_to_cache_dir(var: object, file_name: str, cache_dir: str) -> None:
    file_path = os.path.join(cache_dir, file_name + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved to '{file_path}'")


def load_from_cache_dir(file_name: str, cache_dir: str) -> object:
    file_path = os.path.join(cache_dir, file_name + ".pkl")
    with open(file_path, "rb") as f:
        var = pickle.load(f)
    print(f"'{file_path}' loaded")
    return var

# ========= KL divergence utils
def calculate_KL(p_s: List, q_s: List, est_type: str="basic") -> np.array:
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
        if len(q) == 0:  # for ner - skipping this sample
            kl_list.append(np.nan)
            continue
        
        assert p.size(0) == q.size(0)
        num_y = p.size(0)

        kl_list.append((torch.sum(p - q) / num_y).item())
    return np.array(kl_list)


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
