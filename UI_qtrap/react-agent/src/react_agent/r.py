from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "/home/qtrap/sciborg_dev/notebooks/papers/sb/blanksby_FA.pdf"
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} documents (should match pages in PDF).")
for i, doc in enumerate(documents):
    print(f"\n--- Raw Page {i+1} ---\n{doc.page_content[:1000]}")
