import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_HOME'] = '/tmp/hf_cache'

import gradio as gr
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Load your dataset
df = pd.read_csv("../dataset/dataset1.csv")
df = df.dropna(subset=["Famous Quotes", "Sloka (English Translation)"])
sloka_texts = df["Sloka (English Translation)"].tolist()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Main logic to match quotes
def find_matching_slokas(quote, top_k=3):
    quote_embedding = model.encode(quote, convert_to_tensor=True)
    scores = []
    for i, sloka_text in enumerate(sloka_texts):
        sloka_embedding = model.encode(sloka_text, convert_to_tensor=True)
        score = util.cos_sim(quote_embedding, sloka_embedding)[0][0]
        scores.append((score.item(), i))

    top_scores = sorted(scores, reverse=True)[:top_k]
    results = ""
    for score, idx in top_scores:
        matched = df.iloc[idx]
        results += f"📖 **Sloka:** {matched['Sloka (English Translation)']}\n\n"
        results += f"🔤 **Sanskrit:** {matched.get('Sloka (Sanskrit)', '')}\n"
        results += f"📚 **Interpretation:** {matched.get('Meaning/Interpretation', '')}\n"
        results += f"📌 **Similarity Score:** {round(score, 3)}\n\n---\n\n"
    
    return results

# Gradio UI
demo = gr.Interface(
    fn=find_matching_slokas,
    inputs=[
        gr.Textbox(label="Enter a Motivational Quote", placeholder="e.g., I believe we are here to make the world a better place for all."),
        gr.Slider(1, 5, step=1, label="Top K Slokas")
    ],
    outputs="markdown",
    title="🕉️ Bhagavad Gita Sloka Matcher",
    description="Enter a modern motivational or philosophical quote, and this app will find the most semantically similar Bhagavad Gita slokas."
)

demo.launch(share=True)  