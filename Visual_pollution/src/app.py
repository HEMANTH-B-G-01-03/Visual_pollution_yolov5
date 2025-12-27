import streamlit as st
import os
import subprocess
from PIL import Image
import sys

# ================= PATHS =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_PATH = os.path.join(PROJECT_ROOT, "yolov5")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo", "best.pt")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Visual Pollution Detection", layout="centered")
st.title("🚧 Visual Pollution Detection System")
st.write("Upload an image to detect visual pollution using YOLOv5")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg", "webp"]
)

# ================= HELPER FUNCTION =================
def get_latest_result_folder(base_dir):
    folders = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    return max(folders, key=os.path.getmtime) if folders else None

# ================= MAIN LOGIC =================
if uploaded_file is not None:
    image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", width=700)

    if st.button("Detect Visual Pollution"):
        st.info("Running YOLOv5 detection...")

        command = [
            sys.executable, "detect.py",
            "--weights", MODEL_PATH,
            "--img", "416",
            "--conf", "0.05",
            "--source", image_path,
            "--project", RESULTS_BASE,
            "--name", "frontend"
        ]

        subprocess.run(command, cwd=YOLO_PATH)

        latest_folder = get_latest_result_folder(RESULTS_BASE)

        if latest_folder:
            result_image = os.path.join(latest_folder, uploaded_file.name)

            if os.path.exists(result_image):
                st.success("Detection Completed ✅")
                st.image(result_image, caption="Detection Result", width=700)
            else:
                st.warning("No objects detected in this image.")
        else:
            st.error("Detection output not found.")
