# src/evaluate_bilstm.py
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
df = pd.read_csv("../data/bilstm_sequences.csv")
X = df["class_sequence"].apply(ast.literal_eval)
y = df["label"]  # severity label or final class

# Pad sequences
X = pad_sequences(X, maxlen=20)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
model = load_model("../models/bilstm_model.h5")

# Predict
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
print(classification_report(y_test, y_pred))

# ROC-AUC
auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
print("AUC:", auc)

# Plot Confusion Matrix
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
