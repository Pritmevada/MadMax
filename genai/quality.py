import cv2
import numpy as np

def image_quality_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Sharpness (variance of Laplacian)
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()

    # Normalize (tune later)
    score = 1 / (1 + sharpness)

    return float(score)
