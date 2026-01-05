# import pandas as pd
# import ast
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report,
#     roc_curve,
#     auc,
#     precision_recall_curve
# )

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import to_categorical

# # ================= LOAD DATA =================
# df = pd.read_csv("../data/bilstm_sequences.csv")
# df["class_sequence"] = df["class_sequence"].apply(ast.literal_eval)

# X = df["class_sequence"].tolist()

# # Severity labels
# def severity_label(seq):
#     if len(seq) <= 2:
#         return 0
#     elif len(seq) <= 4:
#         return 1
#     else:
#         return 2

# y = df["class_sequence"].apply(severity_label).values
# y_cat = to_categorical(y, num_classes=3)

# # ================= PREPROCESS =================
# MAX_LEN = 10
# NUM_CLASSES = 11

# X_pad = pad_sequences(X, maxlen=MAX_LEN, padding="post")

# X_train, X_test, y_train, y_test = train_test_split(
#     X_pad, y_cat, test_size=0.2, random_state=42
# )

# # ================= LOAD MODEL =================
# model = load_model("../models/bilstm_severity_model.h5")

# # ================= PREDICTIONS =================
# y_prob = model.predict(X_test)
# y_pred = np.argmax(y_prob, axis=1)
# y_true = np.argmax(y_test, axis=1)

# class_names = ["Low", "Medium", "High"]

# # ================= CONFUSION MATRIX (BLACK STYLE) =================
# cm = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="magma",
#     xticklabels=class_names,
#     yticklabels=class_names
# )
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix – BiLSTM Severity Classification")
# plt.tight_layout()
# plt.savefig("../results/bilstm_confusion_matrix.png")
# plt.show()

# # ================= ROC–AUC CURVE =================
# plt.figure(figsize=(6, 5))

# for i in range(3):
#     fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

# plt.plot([0, 1], [0, 1], "k--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve  BiLSTM Severity Classification")
# plt.legend()
# plt.tight_layout()
# plt.savefig("../results/bilstm_roc_curve.png")
# plt.show()

# # ================= PRECISION–RECALL CURVE =================
# plt.figure(figsize=(6, 5))

# for i in range(3):
#     precision, recall, _ = precision_recall_curve(y_test[:, i], y_prob[:, i])
#     plt.plot(recall, precision, label=class_names[i])

# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision–Recall Curve – BiLSTM")
# plt.legend()
# plt.tight_layout()
# plt.savefig("../results/bilstm_pr_curve.png")
# plt.show()

# # ================= CLASSIFICATION REPORT =================
# report = classification_report(
#     y_true,
#     y_pred,
#     target_names=class_names,
#     output_dict=True
# )

# report_df = pd.DataFrame(report).transpose()
# report_df.to_csv("../results/bilstm_classification_report.csv")

# print("✅ Performance metrics generated successfully")
# print(report_df)




import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score
)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ================= LOAD DATA =================
df = pd.read_csv("../data/bilstm_sequences.csv")
df["class_sequence"] = df["class_sequence"].apply(ast.literal_eval)

X = df["class_sequence"].tolist()

# ---------------- SEVERITY LABELS ----------------
# 0 → Low, 1 → Medium, 2 → High
def severity_label(seq):
    if len(seq) <= 2:
        return 0
    elif len(seq) <= 4:
        return 1
    else:
        return 2

y = df["class_sequence"].apply(severity_label).values
y_cat = to_categorical(y, num_classes=3)

# ================= PREPROCESS =================
MAX_LEN = 10
NUM_CLASSES = 11

X_pad = pad_sequences(X, maxlen=MAX_LEN, padding="post")

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_cat, test_size=0.2, random_state=42
)

# ================= LOAD MODEL =================
model = load_model("../models/bilstm_severity_model.h5")

# ================= PREDICTIONS =================
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

class_names = ["Low", "Medium", "High"]

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="magma",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – BiLSTM Severity Classification")
plt.tight_layout()
plt.savefig("../results/bilstm_confusion_matrix.png", dpi=300)
plt.show()

# ================= ROC–AUC CURVE =================
plt.figure(figsize=(6, 5))

auc_scores = {}

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    auc_scores[class_names[i]] = roc_auc
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – BiLSTM Severity Classification")
plt.legend()
plt.tight_layout()
plt.savefig("../results/bilstm_roc_curve.png", dpi=300)
plt.show()

# ================= PRECISION–RECALL CURVE =================
plt.figure(figsize=(6, 5))

for i in range(3):
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_prob[:, i])
    plt.plot(recall, precision, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – BiLSTM")
plt.legend()
plt.tight_layout()
plt.savefig("../results/bilstm_pr_curve.png", dpi=300)
plt.show()

# ================= CLASSIFICATION REPORT =================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

# ================= ADD AUC VALUES =================
report_df["AUC"] = np.nan
for cls in class_names:
    report_df.loc[cls, "AUC"] = auc_scores[cls]

# Overall AUC scores
macro_auc = roc_auc_score(y_test, y_prob, average="macro")
micro_auc = roc_auc_score(y_test, y_prob, average="micro")

report_df.loc["macro avg", "AUC"] = macro_auc
report_df.loc["micro avg", "AUC"] = micro_auc

# ================= SAVE RESULTS =================
report_df.to_csv("../results/bilstm_classification_report_with_auc.csv")

print("✅ Performance metrics (Accuracy, Precision, Recall, F1, AUC) generated")
print(report_df)
print(f"\nMacro AUC: {macro_auc:.2f}")
print(f"Micro AUC: {micro_auc:.2f}")
