from flask import Flask, request, jsonify, render_template
from rag_pipeline import generate_rag_response, embedding_model, baseline_response
from validation import detect_hallucination
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/baseline", methods=["POST"])
def baseline():
    query = request.json.get("query", "")
    answer = baseline_response(query)
    return jsonify({"answer": answer})

@app.route("/rag", methods=["POST"])
def rag():
    query = request.json.get("query", "")
    answer, docs = generate_rag_response(query, vectorstore)
    return jsonify({"answer": answer})

@app.route("/agritrust", methods=["POST"])
def agritrust():
    query = request.json.get("query", "")
    answer, docs = generate_rag_response(query, vectorstore)
    validation = detect_hallucination(answer, docs)
    return jsonify({
        "answer": answer,
        "validation": validation
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
