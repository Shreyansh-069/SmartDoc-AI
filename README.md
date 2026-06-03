# 🚀 PDF Intelligence Assistant (RAG)

Turn any PDF into a searchable AI knowledge base.

This project uses **Retrieval-Augmented Generation (RAG)** to answer questions directly from uploaded documents. Instead of relying solely on an LLM's memory, it retrieves relevant document chunks, grounds responses in the source material, and provides page citations for transparency.

### ✨ Features

* 📄 Upload and analyze PDF documents
* 🔍 Semantic search with vector embeddings
* 🤖 Context-aware AI responses
* 📚 Source page citations
* 🧠 Retrieval-Augmented Generation (RAG)
* 💬 Conversation history with Streamlit Session State

### 🛠 Tech Stack

* Python
* Streamlit
* LangChain
* FAISS / Vector Store
* Embedding Models
* LLM Integration

### ⚙️ Workflow

```text
PDF → Chunking → Embeddings → Vector Store
                      ↓
User Query → Retrieval → LLM → Answer + Citations
```

### 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 🎯 What I Learned

Building end-to-end RAG pipelines, vector search, metadata handling, source-grounded AI responses, and techniques for improving trust while reducing hallucinations.
