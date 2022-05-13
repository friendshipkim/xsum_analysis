import streamlit as st

# from analysis import storage
from backend.viz_data_model_loader import load_concat_data, load_model_and_tokenizer
from generate_xsum_summary import generate_summaries

# scorer
from utils import score


# import pandas as pd

# cache_summaries = storage.get_summaries("xsum", "facebook-bart-large-xsum")
# cache_keys = cache_summaries.keys()
# annotated_data_by_id = load_annotated_data_by_id()
# filtered_annotated_data_by_id = {
#     k: annotated_data_by_id[k]
#     for k in cache_summaries.keys()
#     if k in annotated_data_by_id
# }

MODEL_NAME = "facebook/bart-large-xsum"

concat_xsum_data = load_concat_data()
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)


def render_model_interface():

    st.title("Generate a Summary with 'facebook-bart-large-xsum' model")

    # Select/Input source document
    selected_id = str(
        st.selectbox("Select entry by bbcid", options=concat_xsum_data.keys())
    )
    selected_data = concat_xsum_data[selected_id]
    document = selected_data["document"]
    st.header("Source Document")
    for x in document.splitlines():
        st.write(x)

    # Perturbed Document
    st.header("Perturbed Document")
    ptb_document = st.text_area("Input:", height=500).replace("\n\n", "\n")
    # for terminal
    # .replace('\\n', '\n').replace("\\", "")
    # st.write("* Perturbed doc == origianl doc?:", ptb_document == document)

    # Ground Truth Summary
    st.header("Summary")
    st.subheader("Ground Truth Summary")
    true_summary = selected_data["true_summary"]
    st.write(true_summary)
    # selected_faithfulness = pd.DataFrame(selected_data["faithfulness_data"])
    # g_summary = (
    #     selected_faithfulness[selected_faithfulness.system == "Gold"].iloc[0].summary
    # )

    # Generated Summary from Original Document
    st.subheader("Generated Summary from Original Document")
    gen_summary, gen_metadata = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        docs_to_summarize=[document],
        num_beams=4,
        return_generation_metadata=True,
    )
    gen_summary = gen_summary[0]  # out of list
    st.write(gen_summary)
    # st.write(gen_metadata)

    # Generated Summary from Perturbed Document
    st.subheader("Generated Summary from Perturbed Document")
    ptb_summary, ptb_metadata = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        docs_to_summarize=[ptb_document],
        num_beams=4,
        return_generation_metadata=True,
    )
    ptb_summary = ptb_summary[0]  # out of list
    st.write(ptb_summary)
    # st.write(ptb_metadata)

    # Evaluation
    st.header("Evaluation")
    st.write("* Perturbed document == original document?:", ptb_document == document)
    st.write("* Perturbed summary == generated summary?:", ptb_summary == gen_summary)

    # bertscore, rouge1

    bert_score_precisions, bert_score_recalls, bert_score_fmeasures = score(
        hyps=[gen_summary, ptb_summary], 
        ref=true_summary, 
        metric="bertscore", 
        model_type="roberta-large"
    )
    rouge1_precisions, rouge1_recalls, rouge1_fmeasures = score(
        hyps=[gen_summary, ptb_summary], 
        ref=true_summary, 
        metric="rouge1"
    )
    rouge2_precisions, rouge2_recalls, rouge2_fmeasures = score(
        hyps=[gen_summary, ptb_summary], 
        ref=true_summary, 
        metric="rouge2"
    )
    rougeL_precisions, rougeL_recalls, rougeL_fmeasures = score(
        hyps=[gen_summary, ptb_summary], 
        ref=true_summary, 
        metric="rougeL"
    )

    st.subheader("F1 rouge1 score")
    st.write("* generated summary:", rouge1_fmeasures[0])
    st.write("* pertrubed summary:", rouge1_fmeasures[1])

    st.subheader("F1 rouge2 score")
    st.write("* generated summary:", rouge2_fmeasures[0])
    st.write("* pertrubed summary:", rouge2_fmeasures[1])

    st.subheader("F1 rougeL score")
    st.write("* generated summary:", rougeL_fmeasures[0])
    st.write("* pertrubed summary:", rougeL_fmeasures[1])

    st.subheader("F1 bert-score using roberta-large")
    st.write("* generated summary:", bert_score_fmeasures[0])
    st.write("* pertrubed summary:", bert_score_fmeasures[1])
    
    # # Output summarization
    # predicted_summary = cache_summaries[selected_id]["summary"][0]
    # st.subheader("Predicted Summary")
    # st.write(predicted_summary)


if __name__ == "__main__":
    render_model_interface()
