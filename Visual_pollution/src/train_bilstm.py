import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bilstm_model import BiLSTM

# ---------------- LOAD DATA ----------------
df = pd.read_csv("../data/bilstm_sequences.csv")

# Convert sequences to list of ints
df["sequence"] = df["sequence"].apply(lambda x: list(map(int, x.split())))

# --------- SIMPLE SEVERITY LABELING (RULE-BASED) ----------
def assign_label(seq):
    if len(seq) >= 6:
        return 2  # High pollution
    elif len(seq) >= 3:
        return 1  # Medium pollution
    else:
        return 0  # Low pollution

df["label"] = df["sequence"].apply(assign_label)

# ---------------- DATASET ----------------
class PollutionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)

    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    return padded, torch.tensor(labels)

dataset = PollutionDataset(df["sequence"].tolist(), df["label"].tolist())
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# ---------------- MODEL ----------------
# Dynamically compute vocab size from data
max_class_id = max(max(seq) for seq in df["sequence"])
vocab_size = max_class_id + 1

model = BiLSTM(vocab_size=vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------- TRAIN ----------------
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "../models/yolo/bilstm.pt")
print("✅ BiLSTM training completed and model saved.")
