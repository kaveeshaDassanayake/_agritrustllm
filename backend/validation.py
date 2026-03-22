_similarity_model = None


def _get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        from sentence_transformers import SentenceTransformer
        _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _similarity_model


def detect_hallucination(answer: str, docs, threshold: float = 0.55):
    from sentence_transformers import util

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not docs or not sentences:
        return {"hallucinated": True, "confidence": 0.0, "unsupported_claims": []}

    model = _get_similarity_model()

    doc_embeddings = model.encode(
        [d.page_content for d in docs], convert_to_tensor=True
    )

    unsupported_claims = []
    similarities = []

    for sentence in sentences:
        sent_emb = model.encode(sentence, convert_to_tensor=True)
        sims = util.cos_sim(sent_emb, doc_embeddings)
        best = float(sims.max())
        similarities.append(best)

        if best < threshold:
            unsupported_claims.append({"claim": sentence, "similarity": round(best, 3)})

    hallucinated = len(unsupported_claims) > 0
    avg_sim = sum(similarities) / len(similarities)
    confidence = round(avg_sim, 2)

    return {
        "hallucinated": hallucinated,
        "confidence": confidence,
        "unsupported_claims": unsupported_claims
    }
