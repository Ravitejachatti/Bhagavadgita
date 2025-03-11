import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load trained model and tokenizer
model_path = "saved_model_v4.0"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load test dataset
df_test = pd.read_csv("./dataset_preparation/slokhas_test.csv")  # Ensure you have a test dataset

# Convert Shlokas to unique numerical labels for mapping back
shloka_mapping = {shloka: idx for idx, shloka in enumerate(df_test["Shloka"].unique())}
reverse_mapping = {idx: shloka for shloka, idx in shloka_mapping.items()}
df_test["Shloka_Label"] = df_test["Shloka"].map(shloka_mapping)

# Tokenize test data
test_encodings = tokenizer(list(df_test["Speech"]), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    predictions = torch.argmax(outputs.logits, dim=1).numpy()

# Convert predictions back to slokas
df_test["Predicted_Sloka_Label"] = predictions
df_test["Predicted_Sloka"] = df_test["Predicted_Sloka_Label"].map(reverse_mapping)

# Compute evaluation metrics
y_true = df_test["Shloka_Label"].values
y_pred = predictions
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save evaluation results
df_test.to_csv("model_predictions_v4.0.csv", index=False)
print("✅ Predictions saved in 'model_predictions_v4.0.csv'")