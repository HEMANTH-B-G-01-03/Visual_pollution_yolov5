import os
import random
import shutil

# Get project root directory (Visual_pollution)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

source_dir = os.path.join(BASE_DIR, "data", "images", "train")
dest_dir = os.path.join(BASE_DIR, "data", "test_images")

os.makedirs(dest_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")
images = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]

if len(images) < 50:
    raise ValueError(f"Only {len(images)} images found, cannot copy 40.")

selected_images = random.sample(images, 40)

for img in selected_images:
    shutil.copy(
        os.path.join(source_dir, img),
        os.path.join(dest_dir, img)
    )

print("✅ 50 random images copied successfully!")
