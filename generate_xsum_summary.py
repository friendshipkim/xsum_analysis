import argparse
import torch
import datasets
import config as cfg
from typing import List, Tuple, Dict
from utils import entropy
from xsum_dataset import XsumDataset

# from sumtool.storage import store_model_summaries
from transformers import BartTokenizer, BartForConditionalGeneration


def load_summarization_model_and_tokenizer(
    model_name: str,
) -> Tuple[BartForConditionalGeneration, BartTokenizer]:
    """
    Load summary generation model and move to GPU, if possible.
    Returns:
        (model, tokenizer)
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(cfg.device)

    return model, tokenizer


def generate_summaries(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    docs_to_summarize: List[str],
    num_beams: int = 4,
    return_generation_metadata: bool = False,
):
    """
    Given a trained summary generation model and appropriate tokenizer,
    1. Tokenize text (and move to device, if possible)
    2. Run inference on model to generate output vocabulary tokens for summary
    3. Decode tokens to a sentence using the tokenizer
    Args:
        model: model to run inference on
        tokenizer: tokenizer corresponding to model
        docs_to_summarize: documents to summarize
        num_beams: number of beams for beam search
        return_generation_metadata: whether generation metadata should be returned
    Returns:
        decoded_sentence
    """
    inputs = tokenizer(
        docs_to_summarize,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    input_token_ids = inputs.input_ids.to(cfg.device)

    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        max_length=cfg.max_summary_length,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]

    if not return_generation_metadata:
        return generated_summaries
    else:
        token_metadata = []
        # bug? - we should check if the token is in **each** input document
        # input_set = input_token_ids.view(-1).tolist()  # this is flattened token ids
        for seq_idx in range(model_output.sequences.shape[0]):
            seq_metadata = []
            token_metadata.append(seq_metadata)

            # save metadata from the second token
            for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
                beam_idx = model_output.beam_indices[seq_idx][idx]

                # get prob distn from score
                selected_beam_probs = torch.exp(model_output.scores[idx][beam_idx])

                # top alternatives during beam search
                beam_top_alternatives = []
                top_probs = torch.topk(selected_beam_probs, k=3)
                for i, v in zip(top_probs.indices, top_probs.values):
                    beam_top_alternatives.append(
                        {
                            "token": tokenizer.decode(i),
                            "token_id": i.item(),
                            "beam_token_prob": v.item(),
                        }
                    )

                seq_metadata.append(
                    {
                        "token_id": output_token_id,
                        "token": tokenizer.decode(output_token_id),
                        "entropy": entropy(
                            selected_beam_probs
                        ),  # entropy of the selected token
                        "beam_token_prob": selected_beam_probs[
                            output_token_id
                        ].item(),  # prob of the selected token
                        "beam_idx": beam_idx.item(),  # beam index of the selected token
                        "beam_top_probs": beam_top_alternatives,  # token, token_id, prob of top K alternatives
                        "token_in_input": output_token_id in input_token_ids[seq_idx],
                        # # is the selected token in its document? (maybe used for overlap)
                        # original - "token_in_input": output_token_id in input_set, bug?
                    }
                )

        return generated_summaries, token_metadata


def generate_token_entropy_metadata(
    bbc_ids: List[str], seq_metadata: List[dict]
) -> Dict[str, Dict]:
    """
    Given a list of bbcids and sequence metadata generated from `generate_summaries`,
    generate token-entropy metadata
    Args:
        bbc_ids: list of bbcids
        seq_metadata: sequence metadata generated from `generate_summaries`
    Returns:
        token_entropy_metadata: dict of {bbcid: tokens_with_entropy},
                                tokens_with_entropy is a list of tuples (token, entropy)
    """
    assert len(bbc_ids) == len(
        seq_metadata
    ), "the number of selected data doesn't match"
    token_entropy_metadata = {}

    for bbc_id, seq_metadata in zip(bbc_ids, seq_metadata):
        tokens_with_entropy = []
        for token_metadata in seq_metadata:
            tokens_with_entropy.append(
                (token_metadata["token"], token_metadata["entropy"])
            )

        token_entropy_metadata[bbc_id] = {"tokens_with_entropy": tokens_with_entropy}
    return token_entropy_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run inference on an xsum example using a pre-trained model"
    )

    parser.add_argument(
        "--bbc_ids",
        type=str,
        required=True,
        help="Comma-separated document BBC IDs in the Xsum dataset",
    )

    # use concatenated data
    # parser.add_argument(
    #     "--data_split",
    #     type=str,
    #     required=True,
    #     choices=["train", "test", "validation"],
    #     help="xsum data split to index into with `data_index`",
    # )

    args = parser.parse_args()

    # load model and tokenizer
    model_name = "facebook/bart-large-xsum"
    model, tokenizer = load_summarization_model_and_tokenizer(model_name)

    # load and concatenate datasets
    xsum_data_raw = datasets.load_dataset("xsum")
    xsum_data_raw_cc = datasets.concatenate_datasets(
        [xsum_data_raw["train"], xsum_data_raw["validation"], xsum_data_raw["test"]]
    )
    xsum_concat_data = XsumDataset(xsum_data_raw_cc)

    # select data
    bbc_ids = [id.strip() for id in args.bbc_ids.split(",")]
    selected_data = [xsum_concat_data.data_by_id[bbc_id] for bbc_id in bbc_ids]
    original_docs = [x["document"] for x in selected_data]

    # generate summary
    gen_summaries, gen_metadata = generate_summaries(
        model, tokenizer, original_docs, num_beams=4, return_generation_metadata=True
    )

    # generate summary metadata
    summary_metadata = generate_token_entropy_metadata(bbc_ids, gen_metadata)

    # print (documents) / summaries / (metadata)
    for source, gen_summary, gen_metadatum in zip(
        selected_data, gen_summaries, gen_metadata
    ):
        print("XSUM ID", source["id"])
        # print("* INPUT DOCUMENT:", source["document"])
        print("* GROUND TRUTH SUMMARY:", source["true_summary"])
        print("* GENERATED SUMMARY:", gen_summary)
        # print("* GENERATION METADATA:", gen_metadatum)
        print()

    # store summaries - TODO
    # store_model_summaries(
    #     "xsum",
    #     model.config.name_or_path,
    #     model.config.to_dict(),
    #     {
    #         source["id"]: gen_summary
    #         for source, gen_summary in zip(selected_data, summaries)
    #     },
    #     summary_metadata
    # )
