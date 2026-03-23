import os
import traceback
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


@app.route("/health")
def health():
    """Simple health check — no ML dependencies."""
    return jsonify({"status": "ok", "python": __import__("sys").version})


@app.route("/")
def home():
    return render_template("index.html")


# MODE A – BASELINE
@app.route("/baseline", methods=["POST"])
def baseline():
    try:
        from rag_pipeline import baseline_response
        query = request.json.get("query", "")
        if not query:
            return jsonify({"answer": "Please provide a query."}), 400
        answer = baseline_response(query)
        return jsonify({"answer": answer})
    except Exception as e:
        err = traceback.format_exc()
        return jsonify({"answer": f"Server error: {str(e)}", "detail": err}), 500


# MODE B – RAG ONLY
@app.route("/rag", methods=["POST"])
def rag():
    try:
        from rag_pipeline import generate_rag_response
        query = request.json.get("query", "")
        if not query:
            return jsonify({"answer": "Please provide a query."}), 400
        answer, docs = generate_rag_response(query, get_vectorstore())
        return jsonify({"answer": answer})
    except Exception as e:
        err = traceback.format_exc()
        return jsonify({"answer": f"Server error: {str(e)}", "detail": err}), 500


# MODE C – RAG + VALIDATION
@app.route("/agritrust", methods=["POST"])
def agritrust():
    try:
        from rag_pipeline import generate_rag_response
        from validation import detect_hallucination
        query = request.json.get("query", "")
        if not query:
            return jsonify({"answer": "Please provide a query."}), 400
        answer, docs = generate_rag_response(query, get_vectorstore())
        validation = detect_hallucination(answer, docs)
        return jsonify({"answer": answer, "validation": validation})
    except Exception as e:
        err = traceback.format_exc()
        return jsonify({"answer": f"Server error: {str(e)}", "detail": err}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
