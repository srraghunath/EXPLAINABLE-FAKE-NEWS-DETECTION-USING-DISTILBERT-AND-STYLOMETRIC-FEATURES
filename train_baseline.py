import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = int(os.getenv("MAX_LEN", "128"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
LR = 2e-5
SUBSAMPLE_SIZE = int(os.getenv("SUBSAMPLE_SIZE", "15000"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

print("Using device:", DEVICE)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# ---------------- DATASET ----------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------- LOAD SOURCE DOMAIN DATA --------
df = pd.read_csv("data/source_domain.csv")
df = df.sample(n=min(SUBSAMPLE_SIZE, len(df)), random_state=42)  # SPEED BOOST

texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

train_texts, _, train_labels, _ = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

train_dataset = NewsDataset(train_texts, train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ---------------- MODEL ------------------
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------- TRAIN ------------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# ------------- SAVE MODEL ----------------
model.save_pretrained("baseline_model")
tokenizer.save_pretrained("baseline_model")

print("FAST BASELINE TRAINING COMPLETE")