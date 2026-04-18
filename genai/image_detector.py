import os
from transformers import pipeline

class ImageAIDetector:
    def __init__(self):
        # Using a model that is well-known for AI vs Real classification
        self.detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")

    def predict(self, image_path):
        results = self.detector(image_path)

        for r in results:
            label = r['label'].lower()
            if "ai" in label or "artificial" in label or "fake" in label:
                return r['score']

        return 0.0
