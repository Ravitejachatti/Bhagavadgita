#####   Testing model with epoch 10 and the 
#####   Model is sentence-transformers/paraphrase-MiniLM-L6-v2

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the saved model and tokenizer
model_path = "saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load dataset to map predictions back to Shlokas
df = pd.read_csv("./dataset_preparation/corrected_dataset.csv")

# Create a mapping from class index to Shloka
shloka_mapping = {idx: shloka for idx, shloka in enumerate(df["Shloka"].unique())}

# Function to predict the most relevant Bhagavad Gita shloka
def predict_shloka(speech_text):
    model.eval()
    inputs = tokenizer(speech_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()  # Convert tensor to integer

    # Return the actual Shloka instead of the index
    return shloka_mapping.get(prediction, "No matching shloka found.")

# Test with an example speech
speech_example = "most weighful think in the world is responsibility"
predicted_shloka = predict_shloka(speech_example)
print("🔹 Predicted Shloka:", predicted_shloka)
