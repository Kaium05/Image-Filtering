import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title(" Fruit Spoilage Estimator")

# Upload the main image (e.g., today's image)
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

# Optional: Upload a previous day's image for histogram comparison
prev_image_file = st.file_uploader("Upload a previous day's fruit image (for comparison)", type=["jpg", "jpeg", "png"])

def load_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

def get_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # --- 1. Color Processing (Discoloration Detection)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown = (10, 50, 50)
    upper_brown = (20, 255, 255)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    color_result = cv2.bitwise_and(image, image, mask=mask)
    st.image(color_result, caption="Discoloration Detection (Color Mask)", use_column_width=True)

    # --- 2. Spatial Filtering (Blur + Edge)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    st.image(edges, caption="Edge Detection (Spatial Filtering)", channels="GRAY", use_column_width=True)

    # --- 3. Histogram Comparison
    if prev_image_file:
        prev_image = load_image(prev_image_file)
        hist1 = get_histogram(image)
        hist2 = get_histogram(prev_image)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        st.image(prev_image, caption="Previous Day's Image", use_column_width=True)
        st.success(f"📊 Histogram Similarity Score: **{similarity:.4f}** (1 = identical, 0 = different)")

    # --- 4. Distance Transform (Spoilage Spread Estimation)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    st.image(dist_norm, caption="Distance Transform (Spoilage Spread)", channels="GRAY", use_column_width=True)
