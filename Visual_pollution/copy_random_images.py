import os
import random
import shutil

IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
TEST_DIR = "data/test_images"

os.makedirs(TEST_DIR, exist_ok=True)

# get images that have labels
valid_images = []
for label in os.listdir(LABEL_DIR):
    if label.endswith(".txt"):
        img_name = label.replace(".txt", ".jpg")
        img_path = os.path.join(IMAGE_DIR, img_name)
        if os.path.exists(img_path):
            valid_images.append(img_name)

# choose random images
NUM_IMAGES = 40   # you can change this
selected = random.sample(valid_images, min(NUM_IMAGES, len(valid_images)))

# copy images
for img in selected:
    src = os.path.join(IMAGE_DIR, img)
    dst = os.path.join(TEST_DIR, img)
    shutil.copy(src, dst)

print(f"✅ Copied {len(selected)} images to data/test_images")
