import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_PATH = "robust_model"
DATA_PATH = "data/target_domain.csv"
MAX_LEN = 96
BATCH_SIZE = 16
TARGET_FRACTION = 0.10   # same as training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------

print("Using device:", DEVICE)

# -------- LOAD MODEL & TOKENIZER --------
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# -------------- DATASET -----------------
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

# -------- LOAD TARGET & SPLIT -----------
df = pd.read_csv(DATA_PATH)

_, target_test = train_test_split(
    df,
    test_size=1 - TARGET_FRACTION,
    stratify=df["label"],
    random_state=42
)

print("Evaluation samples:", len(target_test))
eval_limit = int(os.getenv("EVAL_LIMIT", "0"))
if eval_limit > 0:
    target_test = target_test.head(eval_limit)
    print("Limited evaluation samples:", len(target_test))

texts = target_test["text"].astype(str).tolist()
labels = target_test["label"].tolist()

dataset = NewsDataset(texts, labels)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# -------------- EVALUATION --------------
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating Robust Model"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------- METRICS -----------------------
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary"
)

print("\nROBUST MODEL (UNSEEN TARGET DOMAIN) RESULTS")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")