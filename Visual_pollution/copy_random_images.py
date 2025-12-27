import os
import random
import shutil

# Paths
source_dir = "data/images"
target_dir = "data/test_images"

os.makedirs(target_dir, exist_ok=True)

# Get all image files
images = [f for f in os.listdir(source_dir)
          if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

# Number of images to copy
num_images = 5

# Randomly select images
selected_images = random.sample(images, min(num_images, len(images)))

# Copy images
for img in selected_images:
    src_path = os.path.join(source_dir, img)
    dst_path = os.path.join(target_dir, img)
    shutil.copy(src_path, dst_path)

print("Copied images:")
for img in selected_images:
    print(" -", img)
