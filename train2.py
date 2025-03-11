import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

# ==============================================================
#  📌 STEP 1: LOAD AND PREPROCESS THE DATASET
# ==============================================================
# Load the Bhagavad Gita dataset containing motivational speeches and corresponding slokas.
# The dataset should contain two columns:
# - "Speech": A modern motivational speech excerpt.
# - "Shloka": The most relevant Bhagavad Gita sloka.
df = pd.read_csv("./datset_preparation/corrected_dataset.csv")

# Convert Shlokas to unique numerical labels for classification
shloka_mapping = {shloka: idx for idx, shloka in enumerate(df["Shloka"].unique())}
df["Shloka_Label"] = df["Shloka"].map(shloka_mapping)

# Split dataset into training and validation sets (80-20 split)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Speech"], df["Shloka_Label"], test_size=0.2, random_state=42
)

# ==============================================================
#  📌 STEP 2: TOKENIZATION AND TEXT PROCESSING
# ==============================================================
# Using BERT tokenizer to convert text into input embeddings for the model.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing the training and validation data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

# Convert labels into tensors
train_labels = torch.tensor(train_labels.tolist(), dtype=torch.long)
val_labels = torch.tensor(val_labels.tolist(), dtype=torch.long)

# ==============================================================
#  📌 STEP 3: HANDLE CLASS IMBALANCE WITH WEIGHTED LOSS
# ==============================================================
# Some Shlokas appear more frequently than others in the dataset, causing an imbalance.
# To handle this, we assign higher weights to rare classes.

# Compute class weights (Inverse frequency method)
class_counts = df["Shloka_Label"].value_counts().to_dict()
total_samples = len(df)
class_weights = {label: total_samples / count for label, count in class_counts.items()}

# Convert to tensor format for PyTorch
weights = torch.tensor([class_weights[label] for label in df["Shloka_Label"]], dtype=torch.float)

# Define a sampler to ensure balanced training
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ==============================================================
#  📌 STEP 4: CREATE CUSTOM DATASET CLASS
# ==============================================================
# The dataset class helps format our tokenized inputs into PyTorch tensors.
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

# Create training and validation datasets
train_dataset = BhagavadGitaDataset(train_encodings, train_labels)
val_dataset = BhagavadGitaDataset(val_encodings, val_labels)

# Define DataLoaders with a weighted sampler to handle class imbalance
train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ==============================================================
#  📌 STEP 5: LOAD PRETRAINED MODEL (BERT FOR SEQUENCE CLASSIFICATION)
# ==============================================================
# We use `bert-base-uncased`, a transformer model well-suited for text classification.
# The model is fine-tuned to predict the most relevant Bhagavad Gita sloka.

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=len(shloka_mapping)  # Number of unique Shlokas
)

# Convert computed class weights to tensor format
class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)

# Define loss function with class weights to improve learning on rare classes
loss_function = CrossEntropyLoss(weight=class_weights_tensor)

# ==============================================================
#  📌 STEP 6: DEFINE TRAINING HYPERPARAMETERS
# ==============================================================
# These parameters control the training process:
# - Learning Rate: Adjusted to `1e-5` to allow better convergence.
# - Epochs: Increased to `20` to improve performance over time.
# - Weight Decay: Prevents overfitting.
# - Evaluation Strategy: Evaluates the model after every epoch.
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,  # Lower learning rate for stability
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,  # Increased epochs for better learning
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs"
)

# ==============================================================
#  📌 STEP 7: TRAIN THE MODEL
# ==============================================================
# Hugging Face's Trainer API simplifies the fine-tuning process.

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start model training
trainer.train()

# ==============================================================
#  📌 STEP 8: SAVE TRAINED MODEL (VERSION 2.0)
# ==============================================================
# The trained model and tokenizer are saved for future inference.
# Saving as `saved_model_v2.0` ensures version tracking for improvements.
model.save_pretrained("saved_model_v2.0")
tokenizer.save_pretrained("saved_model_v2.0")

print("✅ Model saved successfully as 'saved_model_v2.0'")