import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Lazy-loaded vectorstore ---
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_community.vectorstores import FAISS
        from rag_pipeline import get_embedding_model
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        _vectorstore = FAISS.load_local(
            os.path.join(BASE_DIR, "faiss_index"),
            get_embedding_model(),
            allow_dangerous_deserialization=True
        )
    return _vectorstore


@app.route("/")
def home():
    return render_template("index.html")


# MODE A – BASELINE
@app.route("/baseline", methods=["POST"])
def baseline():
    from rag_pipeline import baseline_response
    query = request.json.get("query", "")
    answer = baseline_response(query)
    return jsonify({"answer": answer})


# MODE B – RAG ONLY
@app.route("/rag", methods=["POST"])
def rag():
    from rag_pipeline import generate_rag_response
    query = request.json.get("query", "")
    answer, docs = generate_rag_response(query, get_vectorstore())
    return jsonify({"answer": answer})


# MODE C – RAG + VALIDATION
@app.route("/agritrust", methods=["POST"])
def agritrust():
    from rag_pipeline import generate_rag_response
    from validation import detect_hallucination
    query = request.json.get("query", "")
    answer, docs = generate_rag_response(query, get_vectorstore())
    validation = detect_hallucination(answer, docs)
    return jsonify({
        "answer": answer,
        "validation": validation
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
