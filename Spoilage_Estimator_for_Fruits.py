import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🍎 Fruit Spoilage Detector")

uploaded_file = st.file_uploader("Upload an image of a fruit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Uploaded Fruit Image', use_column_width=True)

    # Color processing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown = (10, 50, 50)
    upper_brown = (20, 255, 255)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    result = cv2.bitwise_and(image, image, mask=mask)
    st.image(result, caption='Detected Spoilage (Color-Based)', use_column_width=True)

    # Edge Detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    st.image(edges, caption="Edge Detection", use_column_width=True, channels="GRAY")
