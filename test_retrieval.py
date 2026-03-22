from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the FAISS database you just built
db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)

# Try a test query
query = "What fertilizer is best for tomato plants?"
docs = db.similarity_search(query, k=3)

# Print the top results
for i, d in enumerate(docs, start=1):
    print(f"\nResult {i}:\n{d.page_content}\n")
