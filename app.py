import streamlit as st

from pdf_processor import extract_pdf_text, chunk_documents
from vector_store import create_vector_store, save_vector_store
from rag import ask_question


st.title("📑 PDF analyser and query resolver  (RAG)")

st.write("Upload a PDF and ask questions about it.")


# -------------------
# Upload PDF
# -------------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])


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
query = st.text_input("Enter your question")


if st.button("Get Answer"):

    if not uploaded_file:
        st.warning("Please upload a PDF first.")

    elif not query:
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            result = ask_question(query)

        st.subheader("Answer")
        st.write(result["answer"])

