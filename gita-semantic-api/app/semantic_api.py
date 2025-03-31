# semantic_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
# ✅ Set cache to a writable directory in Hugging Face Spaces
import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_HOME'] = '/tmp/hf_cache'
# Load dataset and model
df = pd.read_csv("./dataset/dataset1.csv")
df = df.dropna(subset=["Famous Quotes", "Sloka (English Translation)"])
model = SentenceTransformer("all-MiniLM-L6-v2")
sloka_texts = df["Sloka (English Translation)"].tolist()
sloka_embeddings = model.encode(sloka_texts, convert_to_tensor=True)

app = FastAPI(title="Bhagavad Gita Semantic Sloka API")

class QuoteRequest(BaseModel):
    quote: str
    top_k: int = 3

@app.post("/match-slokas/")
def find_top_slokas(request: QuoteRequest):
    quote_embedding = model.encode(request.quote, convert_to_tensor=True)

    # Encode slokas on-the-fly (this will be slower but memory-safe)
    scores = []
    for i, sloka_text in enumerate(sloka_texts):
        sloka_embedding = model.encode(sloka_text, convert_to_tensor=True)
        score = util.cos_sim(quote_embedding, sloka_embedding)[0][0]
        scores.append((score.item(), i))

    # Sort by similarity
    top_scores = sorted(scores, reverse=True)[:request.top_k]

    results = []
    for score, idx in top_scores:
        matched = df.iloc[idx]
        results.append({
            "Sloka (English)": matched["Sloka (English Translation)"],
            "Sanskrit": matched.get("Sloka (Sanskrit)", ""),
            "Chapter-Verse": matched.get("Chapter-Verse", ""),
            "Interpretation": matched.get("Meaning/Interpretation", ""),
            "Similarity Score": round(score, 3)
        })

    return {"quote": request.quote, "matches": results}