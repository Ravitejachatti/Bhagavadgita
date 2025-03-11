import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load Bhagavad Gita dataset
df = pd.read_csv("./testing/corrected_dataset.csv")  # Ensure your dataset has "Speech" and "Shloka"

# Convert Shlokas to unique numerical labels
shloka_mapping = {shloka: idx for idx, shloka in enumerate(df["Shloka"].unique())}
df["Shloka_Label"] = df["Shloka"].map(shloka_mapping)
df["Shloka_Label"].value_counts()


# Train-Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Speech"], df["Shloka_Label"], test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

# Convert labels to tensor format
train_labels = torch.tensor(train_labels.tolist(), dtype=torch.long)
val_labels = torch.tensor(val_labels.tolist(), dtype=torch.long)

# Define Custom Dataset
class BhagavadGitaDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]  # Ensure labels are integers
        return item

# Create datasets
train_dataset = BhagavadGitaDataset(train_encodings, train_labels)
val_dataset = BhagavadGitaDataset(val_encodings, val_labels)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2", num_labels=len(df["Shloka"].unique()))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results1",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Save the trained model
model.save_pretrained("saved_model1")
tokenizer.save_pretrained("saved_model1")

print("✅ Model saved successfully in 'saved_model1' directory.")

# Train the model
trainer.train()