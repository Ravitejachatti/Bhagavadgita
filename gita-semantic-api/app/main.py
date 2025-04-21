import os
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

from graph_utils import build_graph, get_knowledge_context
from attention_heatmap import show_attention_heatmap
from visualize_graph import visualize_sloka_graph

# Optional: HuggingFace cache location
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_HOME'] = '/tmp/hf_cache'

# Load dataset
print("🔃 Loading dataset...")
df = pd.read_csv("dataset/dataset1.csv")
df = df.dropna(subset=["Famous Quotes", "Sloka (English Translation)"])
sloka_texts = df["Sloka (English Translation)"].tolist()

# Load powerful semantic model
print("🧠 Loading model...")
model = SentenceTransformer("intfloat/e5-large-v2")

# E5 model requires prefix for better performance
def encode(text):
    return model.encode(f"query: {text}", convert_to_tensor=True)

# Build knowledge graph
print("🌐 Building knowledge graph...")
G = build_graph(df)

# Semantic matching logic
def find_matching_slokas(quote, top_k=3):
    quote_embedding = encode(quote)
    scores = []
    sloka_html_path = None
    heatmap_path = None

    for i, sloka_text in enumerate(sloka_texts):
        sloka_embedding = encode(sloka_text)
        score = util.cos_sim(quote_embedding, sloka_embedding)[0][0]
        scores.append((score.item(), i))

    top_scores = sorted(scores, reverse=True)[:top_k]
    results = ""

    for score, idx in top_scores:
        matched = df.iloc[idx]
        sloka_id = matched.get("Sloka ID", f"Sloka-{idx}")
        context = get_knowledge_context(G, sloka_id)

        results += f"📖 **Sloka:** {matched['Sloka (English Translation)']}\n\n"
        results += f"🔤 **Sanskrit:** {matched.get('Sloka (Sanskrit)', 'N/A')}\n"
        results += f"📚 **Interpretation:** {matched.get('Meaning/Interpretation', 'N/A')}\n"
        results += f"📌 **Similarity Score:** {round(score, 3)}\n"
        results += f"🧠 **Theme:** {context.get('theme', 'Unknown')}\n"
        results += f"📖 Chapter Info: {context.get('chapter', 'Unknown')}\n"
        results += f"🔢 **Verse:** {matched.get('Chapter-Verse', 'N/A')}\n"
        results += f"🔗 **Related Slokas:** {', '.join(context.get('related_slokas', [])[:3])}\n"
        results += "\n---\n\n"

        # 🎯 Visualizations
        print(f"🧠 Generating heatmap for: {quote}")
        heatmap_path = show_attention_heatmap(quote)  # returns saved image path

        print(f"🌐 Generating knowledge graph for: {sloka_id}")
        sloka_html_path = visualize_sloka_graph(sloka_id)  # returns HTML path

        break  # Only show visual for the top result

    return results, sloka_html_path, heatmap_path

# Gradio interface
demo = gr.Interface(
    fn=find_matching_slokas,
    inputs=[
        gr.Textbox(label="Enter a Motivational Quote", placeholder="e.g., You must be the change you wish to see in the world."),
        gr.Slider(1, 5, step=1, label="Top K Slokas")
    ],
    outputs=[
        gr.Markdown(label="Matched Slokas"),
        gr.File(label="Download Knowledge Graph"),
        gr.Image(label="Attention Heatmap")
    ],
    title="🕉️ Bhagavad Gita Sloka Matcher + Knowledge Graph",
    description="Discover the Bhagavad Gita's wisdom based on motivational quotes. Enhanced with knowledge graph context (themes, chapters, related verses)."
)

if __name__ == "__main__":
    demo.launch(share=True)