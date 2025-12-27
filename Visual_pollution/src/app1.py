# import streamlit as st
# import sys
# import os
# from pathlib import Path
# import torch
# import cv2
# import pandas as pd

# # ---------------- PATH SETUP ----------------
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# YOLO_ROOT = PROJECT_ROOT / "yolov5"
# MODEL_YOLO = PROJECT_ROOT / "models" / "yolo" / "best.pt"
# MODEL_BILSTM = PROJECT_ROOT / "models" / "yolo" / "bilstm.pt"
# UPLOAD_DIR = PROJECT_ROOT / "data" / "test_images"

# sys.path.append(str(YOLO_ROOT))
# sys.path.append(str(PROJECT_ROOT / "src"))

# from models.common import DetectMultiBackend
# from utils.general import non_max_suppression
# from utils.augmentations import letterbox
# from utils.torch_utils import select_device
# from bilstm_model import BiLSTM

# # ---------------- STREAMLIT UI ----------------
# st.set_page_config(page_title="Visual Pollution + Severity", layout="centered")
# st.title("🚧 Visual Pollution Detection with Severity (YOLO + BiLSTM)")
# st.write("Upload an image to detect visual pollution and predict severity")

# uploaded_file = st.file_uploader(
#     "Upload an image",
#     type=["jpg", "png", "jpeg", "webp"]
# )

# # ---------------- LOAD YOLO MODEL ----------------
# device = select_device("")
# yolo = DetectMultiBackend(str(MODEL_YOLO), device=device)
# stride = yolo.stride
# imgsz = 416

# # ---------------- LOAD BiLSTM MODEL (FIXED) ----------------
# df = pd.read_csv(PROJECT_ROOT / "data" / "bilstm_sequences.csv")
# df["sequence"] = df["sequence"].apply(lambda x: list(map(int, x.split())))

# max_class_id = max(max(seq) for seq in df["sequence"])
# VOCAB_SIZE = max_class_id + 1

# bilstm = BiLSTM(vocab_size=VOCAB_SIZE)
# bilstm.load_state_dict(torch.load(MODEL_BILSTM, map_location="cpu"))
# bilstm.eval()

# SEVERITY_MAP = {
#     0: "LOW",
#     1: "MEDIUM",
#     2: "HIGH"
# }

# # ---------------- INFERENCE ----------------
# if uploaded_file:
#     UPLOAD_DIR.mkdir(exist_ok=True)
#     img_path = UPLOAD_DIR / uploaded_file.name

#     with open(img_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.image(str(img_path), caption="Uploaded Image", width=500)

#     if st.button("Detect & Predict Severity"):
#         st.info("Running YOLO + BiLSTM pipeline...")

#         # -------- IMAGE PREPROCESS --------
#         img0 = cv2.imread(str(img_path))
#         img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
#         img = img.transpose((2, 0, 1))
#         img = torch.from_numpy(img).float() / 255.0
#         img = img.unsqueeze(0)

#         # -------- YOLO INFERENCE --------
#         with torch.no_grad():
#             preds = yolo(img)
#             preds = non_max_suppression(preds, conf_thres=0.05, iou_thres=0.45)

#         class_sequence = []

#         for det in preds:
#             if det is not None and len(det):
#                 class_sequence = det[:, 5].int().tolist()

#         if not class_sequence:
#             st.warning("No visual pollution detected in this image.")
#         else:
#             st.success(f"Detected class sequence: {class_sequence}")

#             # -------- BiLSTM SEVERITY PREDICTION --------
#             seq_tensor = torch.tensor(class_sequence).unsqueeze(0)

#             with torch.no_grad():
#                 output = bilstm(seq_tensor)
#                 pred = torch.argmax(output, dim=1).item()

#             st.markdown(
#                 f"### 🟢 Predicted Pollution Severity: **{SEVERITY_MAP[pred]}**"
#             )



import streamlit as st
import sys
import os
from pathlib import Path
import torch
import cv2
import pandas as pd

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / "yolov5"
MODEL_YOLO = PROJECT_ROOT / "models" / "yolo" / "best.pt"
MODEL_BILSTM = PROJECT_ROOT / "models" / "yolo" / "bilstm.pt"
UPLOAD_DIR = PROJECT_ROOT / "data" / "test_images"

sys.path.append(str(YOLO_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from bilstm_model import BiLSTM

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Visual Pollution + Severity", layout="centered")
st.title("🚧 Visual Pollution Detection with Severity (YOLO + BiLSTM)")
st.write("Upload an image to detect visual pollution and predict severity")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg", "webp"]
)

# ---------------- LOAD YOLO MODEL ----------------
device = select_device("")
yolo = DetectMultiBackend(str(MODEL_YOLO), device=device)
stride = yolo.stride
imgsz = 416

# ---------------- LOAD BiLSTM MODEL ----------------
df = pd.read_csv(PROJECT_ROOT / "data" / "bilstm_sequences.csv")
df["sequence"] = df["sequence"].apply(lambda x: list(map(int, x.split())))

max_class_id = max(max(seq) for seq in df["sequence"])
VOCAB_SIZE = max_class_id + 1

bilstm = BiLSTM(vocab_size=VOCAB_SIZE)
bilstm.load_state_dict(torch.load(MODEL_BILSTM, map_location="cpu"))
bilstm.eval()

SEVERITY_MAP = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}

# ---------------- INFERENCE ----------------
if uploaded_file:
    UPLOAD_DIR.mkdir(exist_ok=True)
    img_path = UPLOAD_DIR / uploaded_file.name

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Detect & Predict Severity"):
        st.info("Running YOLO + BiLSTM pipeline...")

        # -------- IMAGE PREPROCESS --------
        img0 = cv2.imread(str(img_path))
        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)

        # -------- YOLO INFERENCE --------
        with torch.no_grad():
            preds = yolo(img)
            preds = non_max_suppression(preds, conf_thres=0.05, iou_thres=0.45)

        class_sequence = []
        annotator = Annotator(img0, line_width=2)

        for det in preds:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    class_id = int(cls.item())
                    class_sequence.append(class_id)

                    label = f"class {class_id} {conf:.2f}"
                    annotator.box_label(
                        xyxy,
                        label,
                        color=colors(class_id, True)
                    )

        img0 = annotator.result()

        # -------- DISPLAY IMAGE --------
        st.image(img0, caption="YOLO Detection Output", width=500)

        if not class_sequence:
            st.warning("No visual pollution detected in this image.")
        else:
            st.success(f"Detected class sequence: {class_sequence}")

            # -------- BiLSTM SEVERITY --------
            seq_tensor = torch.tensor(class_sequence).unsqueeze(0)

            with torch.no_grad():
                output = bilstm(seq_tensor)
                pred = torch.argmax(output, dim=1).item()

            st.markdown(
                f"### 🟢 Predicted Pollution Severity: **{SEVERITY_MAP[pred]}**"
            )
