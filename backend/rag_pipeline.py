import os
from dotenv import load_dotenv
load_dotenv()  # loads .env file when running locally

# --- Lazy-loaded globals ---
_embedding_model = None
_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def get_client():
    """Single HuggingFace Inference API client used by all modes."""
    global _client
    if _client is None:
        from huggingface_hub import InferenceClient
        _client = InferenceClient(
            "mistralai/Mistral-7B-Instruct-v0.2",
            token=os.environ.get("HF_API_TOKEN")
        )
    return _client


def _call_api(prompt: str) -> str:
    """Helper to call the HF Inference API and return the answer text."""
    client = get_client()
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    return response.choices[0].message.content.strip()


# Mode A – Baseline: pure LLM answer via API, no retrieval
def baseline_response(query: str) -> str:
    prompt = f"""You are an agricultural advisory assistant for Sri Lanka.
Answer the following question clearly and briefly using your knowledge.

Question: {query}

Answer:"""
    try:
        return _call_api(prompt)
    except Exception as e:
        return f"Error contacting the API: {str(e)}"


# Mode B & C – RAG response via API with retrieved context
def generate_rag_response(query: str, vectorstore, k: int = 4):
    docs = vectorstore.similarity_search_with_score(query, k=8)

    if not docs:
        return "Insufficient information in the knowledge base.", []

    # Filter weak retrievals
    filtered = [d for d, score in docs if score < 1.2]
    if not filtered:
        filtered = [d for d, _ in docs[:3]]
    filtered = filtered[:k]

    context = "\n".join([f"- {doc.page_content.strip()}" for doc in filtered])

    prompt = f"""You are an agricultural advisory assistant for Sri Lanka.

Use ONLY the information given below.
Do not use outside knowledge.
Do not guess.

If the answer is not clearly stated, reply exactly:
Insufficient information in the provided documents.

Context:
{context}

Question:
{query}

Write a short factual answer (1-3 sentences)."""

    try:
        answer = _call_api(prompt)
    except Exception as e:
        return f"Error contacting the API: {str(e)}", filtered

    if "insufficient" in answer.lower() or len(answer.split()) < 5:
        return "Insufficient information in the provided documents.", filtered

    return answer, filtered
