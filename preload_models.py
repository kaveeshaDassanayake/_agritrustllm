#!/usr/bin/env python3
"""
Pre-download all ML models during the build phase so they are cached
before Gunicorn starts, avoiding startup timeouts on Render.
"""
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Pre-downloading sentence-transformers/all-MiniLM-L6-v2 ...")
SentenceTransformer("all-MiniLM-L6-v2")
print("Done.")

print("Pre-downloading google/flan-t5-small ...")
AutoTokenizer.from_pretrained("google/flan-t5-small")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
print("Done.")

print("All models pre-downloaded successfully.")
