import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_PATH = "/home/qtrap/sciborg_dev/notebooks/papers/sb/blanksby_FA.pdf"
VECTOR_DB_PATH = "/home/qtrap/sciborg_dev/faiss_index_blanksby_FA"

# Load the single PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} PDF document(s).")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
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

# Optional: Print the first few chunk contents for inspection
for i, doc in enumerate(filtered_docs[:3]):
    print(f"\n--- Chunk {i} ---\n{doc.page_content[:500]}")
