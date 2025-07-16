import streamlit as st
import gdown
import os
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Google Drive file ID
FILE_ID = "10Vpz4wJPoxsaAXm8tcRkLCiLN6hkfPxB"  # 🔁 Replace with your real one
MODEL_PATH = "best.pt"

# Function to download model from Google Drive
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

# Load model (cached)
model = download_and_load_model()

# App title
st.title("Falcon Safety Detection App 🛡️")
st.write("Upload an image and the model will detect safety-critical equipment.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict
    with st.spinner("Detecting..."):
        results = model.predict(image_rgb, conf=0.25)
        result_img = results[0].plot()

    # Show result
    st.image(result_img, caption="Detection Result", use_column_width=True)
