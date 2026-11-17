import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = int(os.getenv("MAX_LEN", "96"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LR = 2e-5
TARGET_FRACTION = 0.10
SOURCE_SUBSAMPLE = int(os.getenv("SOURCE_SUBSAMPLE", "12000"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------

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
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------- LOAD SOURCE (SUBSAMPLED) --------
source_df = pd.read_csv("data/source_domain.csv")
source_df = source_df.sample(n=min(SOURCE_SUBSAMPLE, len(source_df)), random_state=42)

# -------- LOAD TARGET (10% ONLY) ----------
target_df = pd.read_csv("data/target_domain.csv")

target_train, _ = train_test_split(
    target_df,
    test_size=1 - TARGET_FRACTION,
    stratify=target_df["label"],
    random_state=42
)

# -------- COMBINE TRAIN DATA --------------
train_texts = (
    source_df["text"].astype(str).tolist()
    + target_train["text"].astype(str).tolist()
)
train_labels = (
    source_df["label"].tolist()
    + target_train["label"].tolist()
)

print("Source samples:", len(source_df))
print("Target samples:", len(target_train))
print("Total training:", len(train_texts))

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

# 🔥 FREEZE ENCODER (KEY SPEED BOOST)
for param in model.distilbert.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# ---------------- TRAIN ------------------
model.train()
loop = tqdm(train_loader, desc="Robust Training")

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

    loop.set_postfix(loss=loss.item())

# ------------- SAVE MODEL ----------------
model.save_pretrained("robust_model")
tokenizer.save_pretrained("robust_model")

print("FAST ROBUST TRAINING COMPLETE")