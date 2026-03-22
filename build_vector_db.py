import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents(folder_path):
    documents = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)

                loader = PyPDFLoader(path)
                docs = loader.load()

                documents.extend(docs)

    return documents

#load pdf
docs = load_documents("backend/data/knowledge_base")
print("Total documents loaded:", len(docs))

#Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)
print("Total chunks:", len(chunks))

#Generate embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Build FAISS vector database
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("vector_db")

print("Vector database created successfully")

