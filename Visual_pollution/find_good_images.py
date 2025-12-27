# import os

# label_dir = "data/labels"
# image_dir = "data/images"

# good_images = []

# for file in os.listdir(label_dir):
#     if file.endswith(".txt"):
#         path = os.path.join(label_dir, file)
#         with open(path) as f:
#             lines = [l for l in f if l.strip()]
#             if len(lines) >= 3:   # images with 3+ objects
#                 img_name = file.replace(".txt", ".jpg")
#                 img_path = os.path.join(image_dir, img_name)
#                 if os.path.exists(img_path):
#                     good_images.append(img_name)

# print("Good images found:", len(good_images))
# print("Sample images:")
# for img in good_images[:10]:
#     print(img)




import os
import shutil

LABEL_DIR = "data/labels"
IMAGE_DIR = "data/images"
TEST_DIR = "data/test_images"

os.makedirs(TEST_DIR, exist_ok=True)

MIN_OBJECTS = 3     # images with >= 3 labels
MAX_IMAGES = 30      # number of images to copy

copied = 0

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(LABEL_DIR, label_file)

    with open(label_path, "r") as f:
        lines = [line for line in f if line.strip()]

    if len(lines) >= MIN_OBJECTS:
        img_name_jpg = label_file.replace(".txt", ".jpg")
        img_name_png = label_file.replace(".txt", ".png")

        img_path = None
        if os.path.exists(os.path.join(IMAGE_DIR, img_name_jpg)):
            img_path = os.path.join(IMAGE_DIR, img_name_jpg)
            img_name = img_name_jpg
        elif os.path.exists(os.path.join(IMAGE_DIR, img_name_png)):
            img_path = os.path.join(IMAGE_DIR, img_name_png)
            img_name = img_name_png

        if img_path:
            shutil.copy(img_path, os.path.join(TEST_DIR, img_name))
            print("Copied:", img_name)
            copied += 1

        if copied >= MAX_IMAGES:
            break

print(f"\n✅ Done! {copied} easy images copied to data/test_images/")
