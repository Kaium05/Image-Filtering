import cv2
import numpy as np
import os

def mood_enhancer(img_gray):
    # Histogram Equalization
    hist_eq = cv2.equalizeHist(img_gray)

    # Gaussian Smoothing
    blur = cv2.GaussianBlur(hist_eq, (5, 5), 0)

    # Sharpening Kernel
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5,-1],
                             [0, -1, 0]])
    sharpened = cv2.filter2D(blur, -1, kernel_sharp)
    return sharpened

def poor_lighting_fixer(img_color):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Detect dark areas
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Histogram Equalization on Y channel
    ycrcb = cv2.cvtColor(img_color, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # Blend enhanced parts into original
    mask_colored = cv2.merge([mask, mask, mask])
    output = np.where(mask_colored == 255, img_eq, img_color)
    return output

def main():
    print("=== Image Processing Project ===")
    path = input("Enter the image path: ").strip()
    
    if not os.path.isfile(path):
        print("❌ File not found! Please check the path.")
        return

    choice = input("Choose project: [1] Mood Enhancer | [2] Poor Lighting Fixer: ").strip()
    
    if choice == '1':
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print("❌ Failed to load image.")
            return
        enhanced = mood_enhancer(img_gray)
        cv2.imshow("Original (Grayscale)", img_gray)
        cv2.imshow("Mood Enhanced", enhanced)

    elif choice == '2':
        img_color = cv2.imread(path)
        if img_color is None:
            print("❌ Failed to load image.")
            return
        enhanced = poor_lighting_fixer(img_color)
        cv2.imshow("Original", img_color)
        cv2.imshow("Low Light Enhanced", enhanced)

    else:
        print("❌ Invalid choice! Please enter 1 or 2.")
        return

    print("✅ Press any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
