from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_pdf_text(pdf_file):
    #extract text from pdf and attach to meta data

    reader = PdfReader(pdf_file)
    documents = []

    for page_num , page in enumerate(reader.pages , start = 1):

        text = page.extract_text()

        if text and text.strip():

            documents.append(
                Document(
                    page_content = text,
                    metadata = {
                        "page" :page_num
                    }
                )
            )
            
    return documents

def chunk_documents(documents):
    # split documents into chunks 

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(documents)
    
    return chunks


