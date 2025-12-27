import sys
import os
import csv
from pathlib import Path

# Add yolov5 to path
ROOT = Path(__file__).resolve().parents[1] / "yolov5"
sys.path.append(str(ROOT))

import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# ---------------- PATHS ----------------
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "yolo" / "best.pt")
IMAGE_DIR = str(Path(__file__).resolve().parents[1] / "data" / "images")
OUTPUT_CSV = str(Path(__file__).resolve().parents[1] / "data" / "bilstm_sequences.csv")

# ---------------- LOAD MODEL ----------------
device = select_device("")
model = DetectMultiBackend(MODEL_PATH, device=device)
stride = model.stride
imgsz = 416

sequences = []

# ---------------- PROCESS IMAGES ----------------
for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)

    img0 = cv2.imread(img_path)
    if img0 is None:
        continue

    # YOLO preprocessing (VERY IMPORTANT)
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))  # HWC → CHW
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.05, iou_thres=0.45)

    for det in pred:
        if det is not None and len(det):
            classes = det[:, 5].int().tolist()
            sequences.append([img_name, " ".join(map(str, classes))])

# ---------------- SAVE CSV ----------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "sequence"])
    writer.writerows(sequences)

print("✅ YOLO sequences extracted successfully!")
print(f"Saved to: {OUTPUT_CSV}")
