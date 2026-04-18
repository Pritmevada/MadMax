import cv2
import numpy as np

def noise_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return 0.0

    noise = img.astype(float) - cv2.GaussianBlur(img, (5,5), 0).astype(float)
    variance = np.var(noise)

    # Normalize to 0-1 using sigmoid-like scaling
    # High variance (>200) → score near 1.0, low variance (<20) → near 0.0
    score = 1.0 / (1.0 + np.exp(-0.02 * (variance - 100)))

    return float(score)
