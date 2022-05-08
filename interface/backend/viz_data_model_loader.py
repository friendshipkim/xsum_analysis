import streamlit as st
import datasets
from xsum_dataset import XsumDataset
from generate_xsum_summary_from_input import load_summarization_model_and_tokenizer

# def prepare_data_for_view(data):
#     factuality_data = {}
#     for system in data["factuality_data"].keys():
#         labels = data["factuality_data"][system]["labels"]
#         factuality_data[system] = data["factuality_data"][system]
#         factuality_data[system]["mean_worker_factuality_score"] = sum(
#             labels.values()
#         ) / len(labels)

#     flat_faithfulness_data = []
#     for val in data["faithfulness_data"].values():
#         for label in val["labels"]:
#             flattened = {
#                 "system": val["system"],
#                 "summary": val["summary"],
#             }
#             for label_key, label_value in label.items():
#                 flattened[label_key] = label_value
#             flat_faithfulness_data.append(flattened)

#     return {
#         "document": data["document"],
#         "factuality_data": factuality_data,
#         "faithfulness_data": flat_faithfulness_data,
#     }


@st.experimental_memo
def load_concat_data():
    xsum_data_raw = datasets.load_dataset("xsum")
    xsum_data_raw_cc = datasets.concatenate_datasets(
        [xsum_data_raw["train"], xsum_data_raw["validation"], xsum_data_raw["test"]]
        )
    xsum_concat_data = XsumDataset(xsum_data_raw_cc)
    # view_dataset = {
    #     k: prepare_data_for_view(v)
    #     for k, v in dataset.data_by_id.items()
    #     if len(v["faithfulness_data"]) > 0
    # }
    # return view_dataset
    view_data = xsum_concat_data.data_by_id
    return view_data

@st.experimental_memo
def load_model_and_tokenizer(model_name):
    return load_summarization_model_and_tokenizer(model_name)
