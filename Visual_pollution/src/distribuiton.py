# import os
# import pandas as pd
# from collections import defaultdict

# # ===== PATHS =====
# BASE_PATH = "../data"
# SPLITS = ["train", "val", "test"]  # use only train & val if test not present
# LABEL_DIR = "labels"

# # ===== CLASS NAMES (update if needed) =====
# class_names = [
#     "garbage",
#     "posters",
#     "poles",
#     "banners",
#     "construction_barrier",
#     "broken_signboard",
#     "potholes",
#     "clutter_sidewalk",
#     "road_damage",
#     "graffiti",
#     "other"
# ]

# # ===== INIT COUNTERS =====
# counts = {
#     "Class Label": class_names,
#     "Training Images": [0]*len(class_names),
#     "Validation Images": [0]*len(class_names),
#     "Testing Images": [0]*len(class_names)
# }

# # ===== COUNT FUNCTION =====
# def count_labels(split):
#     label_path = os.path.join(BASE_PATH, LABEL_DIR, split)
#     class_counter = defaultdict(set)

#     if not os.path.exists(label_path):
#         return class_counter

#     for file in os.listdir(label_path):
#         if not file.endswith(".txt"):
#             continue

#         img_id = file.replace(".txt", "")
#         with open(os.path.join(label_path, file), "r") as f:
#             for line in f:
#                 cls = int(line.split()[0])
#                 class_counter[cls].add(img_id)

#     return class_counter

# #testing image number 

# def count_test_images():
#     test_image_dir = os.path.join(BASE_PATH, "test_images")
#     if not os.path.exists(test_image_dir):
#         return 0
#     return len([
#         f for f in os.listdir(test_image_dir)
#         if f.lower().endswith((".jpg", ".png", ".jpeg"))
#     ])


# # ===== PROCESS SPLITS =====
# train_counts = count_labels("train")
# val_counts = count_labels("val")
# test_counts = count_labels("test")

# for i in range(len(class_names)):
#     counts["Training Images"][i] = len(train_counts.get(i, []))
#     counts["Validation Images"][i] = len(val_counts.get(i, []))
#     counts["Testing Images"][i] = len(test_counts.get(i, []))

# # ===== CREATE TABLE =====
# df = pd.DataFrame(counts)

# # ===== SAVE =====
# df.to_csv("../results/class_distribution_table.csv", index=False)
# df.to_excel("../results/class_distribution_table.xlsx", index=False)

# print("✅ Class distribution table generated")
# print(df)



import os
import pandas as pd
from collections import defaultdict
import matplotlib as plt 
# ===== PATHS =====
BASE_PATH = "../data"

TRAIN_LABEL_PATH = os.path.join(BASE_PATH, "labels", "train")
VAL_LABEL_PATH   = os.path.join(BASE_PATH, "labels", "val")

# 🔥 AUTO-LABEL TEST PATHS
TEST_LABEL_PATH  = os.path.join(BASE_PATH, "auto_labels", "labels", "labels")
TEST_IMAGE_PATH  = os.path.join(BASE_PATH, "auto_labels", "labels", "images")

# ===== CLASS NAMES =====
class_names = [
    "garbage",
    "posters",
    "poles",
    "banners",
    "construction_barrier",
    "broken_signboard",
    "potholes",
    "clutter_sidewalk",
    "road_damage",
    "graffiti",
    "other"
]

# ===== INIT COUNTERS =====
counts = {
    "Class Label": class_names,
    "Training Images": [0]*len(class_names),
    "Validation Images": [0]*len(class_names),
    "Testing Images": [0]*len(class_names)
}

# ===== COUNT LABELS FUNCTION =====
def count_labels(label_path):
    class_counter = defaultdict(set)

    if not os.path.exists(label_path):
        return class_counter

    for file in os.listdir(label_path):
        if not file.endswith(".txt"):
            continue

        img_id = file.replace(".txt", "")
        with open(os.path.join(label_path, file), "r") as f:
            for line in f:
                cls = int(line.split()[0])
                class_counter[cls].add(img_id)

    return class_counter

# ===== COUNT TEST IMAGES =====
def count_test_images():
    if not os.path.exists(TEST_IMAGE_PATH):
        return 0
    return len([
        f for f in os.listdir(TEST_IMAGE_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

# ===== PROCESS =====
train_counts = count_labels(TRAIN_LABEL_PATH)
val_counts   = count_labels(VAL_LABEL_PATH)
test_counts  = count_labels(TEST_LABEL_PATH)

for i in range(len(class_names)):
    counts["Training Images"][i]   = len(train_counts.get(i, []))
    counts["Validation Images"][i] = len(val_counts.get(i, []))
    counts["Testing Images"][i]    = len(test_counts.get(i, []))

# ===== CREATE TABLE =====
df = pd.DataFrame(counts)

# ===== SAVE =====
os.makedirs("../results", exist_ok=True)
df.to_csv("../results/class_distribution_table.csv", index=False)
df.to_excel("../results/class_distribution_table.xlsx", index=False)

print("✅ Class distribution table generated successfully")
print(df)


# ===== SAVE AS JPG IMAGE =====
plt.figure(figsize=(14, 6))
plt.axis('off')

table = plt.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

plt.title("Class-wise Dataset Distribution", fontsize=14, pad=20)

plt.savefig("../results/class_distribution_table.jpg", dpi=300, bbox_inches='tight')
plt.close()

print("🖼️ JPG image saved successfully")
