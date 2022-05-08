import streamlit as st
from model_interface import render_model_interface


def render():
    pages = {
        "Model Interface": render_model_interface,
    }

    st.sidebar.title("Xsum Analysis")
    selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

    pages[selected_page]()


if __name__ == "__main__":
    render()
