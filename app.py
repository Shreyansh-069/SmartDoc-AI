import streamlit as st

from pdf_processor import extract_pdf_text, chunk_documents
from vector_store import create_vector_store, save_vector_store
from rag import ask_question


st.set_page_config(
    page_title="SmartDoc-AI",
    page_icon="📄",
    layout="wide"
)

# Header
st.title("📄 SmartDoc-AI")
st.caption("AI-Powered PDF Question Answering System using RAG")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        """
        Upload one or more PDF files, process them,
        and ask questions about their contents using
        Retrieval-Augmented Generation (RAG).
        """
    )

    st.info(
        "Workflow:\n"
        "PDFs → Chunks → Embeddings → FAISS → Llama3"
    )

# Main Layout
col1, col2 = st.columns([1, 1])

# -----------------------------
# PDF Upload Section
# -----------------------------
with col1:

    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"{len(uploaded_files)} PDF(s) uploaded")

        st.write("### Uploaded Files")
        for pdf in uploaded_files:
            st.write(f"• {pdf.name}")

        if st.button("⚙️ Process PDFs", use_container_width=True):

            try:
                with st.spinner("Processing PDFs..."):

                    all_documents = []

                    for pdf in uploaded_files:
                        docs = extract_pdf_text(pdf)

                        # extract_pdf_text should return a list of Documents
                        all_documents.extend(docs)

                    chunks = chunk_documents(all_documents)

                    vector_store = create_vector_store(chunks)

                    save_vector_store(vector_store)

                st.success(
                    f"Successfully processed {len(uploaded_files)} PDF(s)!"
                )

            except Exception as e:
                st.error(f"Error while processing PDFs: {e}")

# -----------------------------
# Question Answering Section
# -----------------------------
with col2:

    st.subheader("💬 Ask Questions")

    query = st.text_input(
        "Enter your question"
    )

    if st.button("🚀 Get Answer", use_container_width=True):

        if not query:
            st.warning("Please enter a question.")

        else:

            try:
                with st.spinner("Generating answer..."):

                    result = ask_question(query)

                st.success("Answer Generated")

                st.markdown("### 📝 Answer")
                st.write(result["answer"])

                st.markdown("### 📚 Source Pages")

                sources = result["sources"]

                formatted_sources = []
                for source in sources:
                    if "(page" in source:
                        pdf_name, page = source.split("(page")
                        page = page.replace(")", "").strip()

                        formatted_sources.append(
                        f"📄 {pdf_name.strip()} | Page {page}\n"
                        )

                st.info("\n".join(formatted_sources))

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

st.caption(
    "Built with Streamlit • LangChain • FAISS • BGE Embeddings • Llama3"
)