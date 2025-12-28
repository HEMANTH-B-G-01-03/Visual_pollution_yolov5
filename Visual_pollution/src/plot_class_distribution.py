# import pandas as pd
# import matplotlib.pyplot as plt

# # Load YOLO predictions log (from detect output)
# data = {
#     "class": ["class3", "class4", "class9", "class7", "class2"],
#     "count": [420, 310, 290, 180, 95]  # example counts
# }

# df = pd.DataFrame(data)

# plt.figure(figsize=(8,5))
# plt.bar(df["class"], df["count"])
# plt.xlabel("Visual Pollution Classes")
# plt.ylabel("Detection Count")
# plt.title("Class-wise Visual Pollution Detection")
# plt.show()


import os
from collections import Counter
import matplotlib.pyplot as plt

# 🔁 CHANGE THIS to your actual labels folder
LABELS_DIR = "../data/labels/"

class_counts = Counter()

# Read label files
for file in os.listdir(LABELS_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(LABELS_DIR, file), "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

# Sort by class id
classes = sorted(class_counts.keys())
counts = [class_counts[c] for c in classes]

# Plot
plt.figure()
plt.bar(classes, counts)
plt.xlabel("Class ID")
plt.ylabel("Number of Annotations")
plt.title("Class Distribution in Dataset")
plt.xticks(classes)
plt.show()
