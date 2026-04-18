from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIPDetector:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=[
                "a real photograph",
                "an AI generated image",
                "a fake image",
                "a synthetic image"
            ],
            images=image,
            return_tensors="pt",
            padding=True
        )

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        # Combine AI-related probabilities
        ai_score = float(probs[0][1] + probs[0][2] + probs[0][3])
        return ai_score
