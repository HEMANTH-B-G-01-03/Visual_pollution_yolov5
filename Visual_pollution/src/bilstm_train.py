
# # all Confusion Matrix, Precision, Recall, F1, AUC

# import pandas as pd
# import ast
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt

# # ================= LOAD DATA =================
# df = pd.read_csv("../data/bilstm_sequences.csv")

# # Convert string "[4,4,4]" → list
# df["class_sequence"] = df["class_sequence"].apply(ast.literal_eval)

# X = df["class_sequence"].tolist()

# # ---------------- LABEL CREATION ----------------
# # Simple severity rule (for demo + paper)
# # 1–2 objects → Low
# # 3–4 objects → Medium
# # ≥5 objects → High
# def severity_label(seq):
#     if len(seq) <= 2:
#         return 0  # Low
#     elif len(seq) <= 4:
#         return 1  # Medium
#     else:
#         return 2  # High

# y = df["class_sequence"].apply(severity_label).values
# y = to_categorical(y, num_classes=3)

# # ================= SEQUENCE PROCESSING =================
# MAX_LEN = 10   # max sequence length
# NUM_CLASSES = 11  # YOLO classes (0–10)

# X_pad = pad_sequences(
#     X,
#     maxlen=MAX_LEN,
#     padding="post",
#     truncating="post"
# )

# X_train, X_val, y_train, y_val = train_test_split(
#     X_pad, y, test_size=0.2, random_state=42
# )

# # ================= BiLSTM MODEL =================
# model = Sequential([
#     Embedding(
#         input_dim=NUM_CLASSES,
#         output_dim=64,
#         input_length=MAX_LEN
#     ),
#     Bidirectional(LSTM(64, return_sequences=True)),
#     Dropout(0.3),
#     Bidirectional(LSTM(32)),
#     Dense(32, activation="relu"),
#     Dense(3, activation="softmax")
# ])

# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # ================= TRAIN =================
# model.summary()

# model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=20,
#     batch_size=32
# )

# # ================= SAVE =================
# model.save("../models/bilstm_severity_model.h5")
# print("✅ BiLSTM model trained and saved")

# model.build(input_shape=(None, MAX_LEN))
# model.summary()

# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training and Validation Accuracy")
# plt.show()

# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.show()



# ================= IMPORTS =================
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

#  LOAD DATA 
df = pd.read_csv("../data/bilstm_sequences.csv")

# Convert "[4,4,4]" → [4,4,4]
df["class_sequence"] = df["class_sequence"].apply(ast.literal_eval)

X = df["class_sequence"].tolist()

#  SEVERITY LABELS 
# 0 → Low, 1 → Medium, 2 → High
def severity_label(seq):
    if len(seq) <= 2:
        return 0
    elif len(seq) <= 4:
        return 1
    else:
        return 2

y = df["class_sequence"].apply(severity_label).values
y = to_categorical(y, num_classes=3)

# SEQUENCE PADDING 
MAX_LEN = 10
NUM_CLASSES = 11  # YOLO classes (0–10)

X_pad = pad_sequences(
    X,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_train, X_val, y_train, y_val = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

#  BiLSTM MODEL 
model = Sequential([
    Embedding(input_dim=NUM_CLASSES, output_dim=64),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#  TRAIN  
history = model.fit(   # ✅ store history
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# SAVE MODEL 
model.save("../models/bilstm_severity_model.h5")
print("✅ BiLSTM model trained and saved")

# ACCURACY PLOT 
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.show()

# LOSS PLOT 
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.show()


