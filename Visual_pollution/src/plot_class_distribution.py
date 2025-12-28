import pandas as pd
import matplotlib.pyplot as plt

# Load YOLO predictions log (from detect output)
data = {
    "class": ["class3", "class4", "class9", "class7", "class2"],
    "count": [420, 310, 290, 180, 95]  # example counts
}

df = pd.DataFrame(data)

plt.figure(figsize=(8,5))
plt.bar(df["class"], df["count"])
plt.xlabel("Visual Pollution Classes")
plt.ylabel("Detection Count")
plt.title("Class-wise Visual Pollution Detection")
plt.show()
