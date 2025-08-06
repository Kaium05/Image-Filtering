import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to get image path from user and check if it exists
def get_image_from_user():
    image_path = input("Enter the path to the fruit image: ").strip()
    if not os.path.exists(image_path):
        print("Image not found. Please check the path.")
        return None
    return image_path

# Function to load image
def load_image(image_path):
    return cv2.imread(image_path)

# Function to show image using matplotlib (for better display)
def show_image(title, img, cmap=None):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if cmap is None else img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Color Image Processing: highlight discoloration/mold
def color_processing(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown = (10, 50, 50)
    upper_brown = (20, 255, 255)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Spatial Filtering: blur and detect edges
def spatial_filtering(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges

# Histogram calculation
def get_histogram(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Histogram comparison
def compare_histograms(img1_path, img2_path):
    hist1 = get_histogram(img1_path)
    hist2 = get_histogram(img2_path)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Distance Transform: estimate spread of spoilage
def distance_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    return dist

# Main pipeline
image_path = get_image_from_user()
if image_path:
    image = load_image(image_path)
    show_image("Original Image", image)

    # Color processing
    color_result = color_processing(image)
    show_image("Discoloration Detection", color_result)

    # Spatial filtering
    edge_result = spatial_filtering(image)
    show_image("Edge Detection", edge_result, cmap='gray')

    # Distance transform
    dist_result = distance_transform(image)
    show_image("Distance Transform (Spoilage Spread)", dist_result, cmap='gray')

    # Optional histogram comparison (if comparing with previous image)
    compare_choice = input("Do you want to compare with a previous image? (yes/no): ").strip().lower()
    if compare_choice == 'yes':
        prev_image_path = input("Enter previous image path: ").strip()
        if os.path.exists(prev_image_path):
            similarity = compare_histograms(prev_image_path, image_path)
            print(f"Histogram Similarity (1 = same, 0 = different): {similarity:.4f}")
        else:
            print("Previous image not found. Skipping histogram comparison.")
