from transformers import pipeline

baseline_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

def baseline_response(query):

    prompt = f"""
Answer the following agricultural question clearly and briefly.

Question:
{query}

Answer:
"""

    output = baseline_llm(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        repetition_penalty=1.2
    )

    return output[0]["generated_text"].strip()

