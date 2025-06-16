import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


PDF_DIR = "/home/qtrap/sciborg_dev/notebooks/papers/qtrap_nano"
VECTOR_DB_PATH = "/home/qtrap/sciborg_dev/faiss_index_qtrap_nano"

loader = DirectoryLoader(PDF_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} PDF documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} text chunks.")

# Optional: Filter out front-matter/metadata chunks
filtered_docs = [
    doc for doc in docs
    if len(doc.page_content) > 100 and
       not any(word in doc.page_content.lower() for word in ["author", "reprints", "permissions", "copyright", "doi", "journal", "published"])
]
print(f"Filtered down to {len(filtered_docs)} chunks.")

embeddings = OpenAIEmbeddings(openai_api_key="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")
vectordb = FAISS.from_documents(filtered_docs, embeddings)
vectordb.save_local(VECTOR_DB_PATH)
print(f"Saved FAISS index to {VECTOR_DB_PATH}")

