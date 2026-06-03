import streamlit as st

from pdf_processor import extract_pdf_text, chunk_documents
from vector_store import create_vector_store, save_vector_store
from rag import ask_question


# -------------------
# Page Config
# -------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📑",
    layout="centered"
)

st.title("📑 PDF Analyser and Query Resolver (RAG)")
st.write("Upload a PDF and ask questions about it.")


# -------------------
# Session State
# -------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------
# Upload PDF
# -------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# -------------------
# Process PDF
# -------------------

if uploaded_file:

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            docs = extract_pdf_text(uploaded_file)
            chunks = chunk_documents(docs)

            vector_store = create_vector_store(chunks)
            save_vector_store(vector_store)

        st.success("PDF processed successfully!")


# -------------------
# Ask Question
# -------------------

query = st.text_input(
    "Enter your question"
)

if st.button("Get Answer"):

    if not uploaded_file:
        st.warning("Please upload a PDF first.")

    elif not query:
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            result = ask_question(query)

        # Save conversation
        st.session_state.history.append(
            {
                "question": query,
                "answer": result["answer"],
                "sources": result["sources"]
            }
        )


# -------------------
# Conversation History
# -------------------

if st.session_state.history:

    st.divider()
    st.subheader("Conversation History")

    for i, item in enumerate(
        reversed(st.session_state.history),
        start=1
    ):

        with st.container(border=True):

            st.markdown(
                f"### Question {len(st.session_state.history) - i + 1}"
            )

            st.write(item["question"])

            st.markdown("### Answer")

            st.write(item["answer"])

            st.markdown("### Source Pages")

            if item["sources"]:

                pages = sorted(set(item["sources"]))

                badges = " ".join(
                    [f"`Page {page}`" for page in pages]
                )

                st.markdown(badges)

            else:
                st.info("No source pages available.")