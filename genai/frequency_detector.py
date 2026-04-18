import cv2
import numpy as np

def frequency_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return 0.0

    # Fourier Transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # Measure irregularity
    mean = np.mean(magnitude)
    std = np.std(magnitude)

    ratio = std / (mean + 1e-5)

    # Normalize to 0-1 using sigmoid-like scaling
    # Typical ratio range is 0.2-0.6, map to 0-1
    score = 1.0 / (1.0 + np.exp(-10 * (ratio - 0.4)))

    return float(score)
