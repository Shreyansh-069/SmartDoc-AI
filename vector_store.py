from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

def get_embedding_model():
    #load embedding model from hugging face

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}

    )

def create_vector_store(chunks):
    #create FAISS index from chunks 

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding=embeddings
    )

    
    return vector_store

def save_vector_store(vector_store):
    #save FAISS index to local folder 

    vector_store.save_local("vector_db")


def load_vector_store():

    #load saved FAISS index 

    embeddings = get_embedding_model()

    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization = True
    )

    return vector_store

    