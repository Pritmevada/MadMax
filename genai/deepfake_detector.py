from transformers import pipeline

class DeepFakeDetector:
    def __init__(self):
        self.detector = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

    def predict(self, image_path):
        results = self.detector(image_path)

        for r in results:
            label = r['label'].lower()
            if "fake" in label or "deepfake" in label or "ai" in label or "synthetic" in label:
                return r['score']

        return 0.0
