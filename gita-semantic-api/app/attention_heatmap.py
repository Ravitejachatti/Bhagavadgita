import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel


def show_attention_heatmap(sentence, save_path="attention_heatmap.png"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)

    # Get average attention from last layer
    attentions = outputs.attentions[-1][0]
    avg_attention = attentions.mean(dim=0).detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu")
    plt.title("Attention Heatmap - Word Importance")

    # ✅ Save instead of showing
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return save_path  # Return path so it can be used in Gradio or CLI