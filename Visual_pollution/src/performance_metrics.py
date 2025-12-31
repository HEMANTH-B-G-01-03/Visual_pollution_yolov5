import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import seaborn as sns

# ================= LOAD DATA =================
df = pd.read_csv("../data/bilstm_sequences.csv")
df["class_sequence"] = df["class_sequence"].apply(ast.literal_eval)

X = df["class_sequence"].tolist()

def severity_label(seq):
    if len(seq) <= 2:
        return 0  # Low
    elif len(seq) <= 4:
        return 1  # Medium
    else:
        return 2  # High

y = df["class_sequence"].apply(severity_label).values
y_cat = to_categorical(y, num_classes=3)

# ================= PAD SEQUENCES =================
MAX_LEN = 10
X_pad = pad_sequences(X, maxlen=MAX_LEN, padding="post")

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_cat, test_size=0.2, random_state=42
)

# ================= LOAD TRAINED MODEL =================
model = load_model("../models/bilstm_severity_model.h5")

# ================= PREDICTION =================
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BiLSTM Severity Classification")
plt.show()

# ================= PRECISION, RECALL, F1 =================
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Low", "Medium", "High"]
))

# ================= ROC & AUC =================
plt.figure(figsize=(6, 5))
for i, label in enumerate(["Low", "Medium", "High"]):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - BiLSTM Severity Classification")
plt.legend()
plt.show()
