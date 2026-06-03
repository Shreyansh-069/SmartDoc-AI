from langchain_ollama import ChatOllama
from vector_store import load_vector_store


MODEL_NAME ="llama3"

def get_llm():
   
   # Load Ollama Llama3.
    

    return ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

def get_relevant_documents(query, k=4):
    
    #Retrieve top-k relevant chunks.

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    return docs


def build_context(documents):
    
    #Convert retrieved documents into context string.
    

    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    return context

def ask_question(query):
    
    #Main RAG function.
    

    docs = get_relevant_documents(query)

    context = build_context(docs)

    prompt = f"""
You are a helpful PDF assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
reply:

"I could not find this information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    source_pages = sorted(
        set(
            doc.metadata.get("page")
            for doc in docs
        )
    )

    return {
        "answer": response.content,
        "sources": source_pages
    }