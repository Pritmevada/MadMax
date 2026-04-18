from image_detector import ImageAIDetector
from deepfake_detector import DeepFakeDetector
from clip_detector import CLIPDetector
from quality import image_quality_score
from frequency_detector import frequency_score
from noise import noise_score

class AdvancedDetector:
    def __init__(self):
        self.model_ai = ImageAIDetector()       # umm-maybe/AI-image-detector
        self.model_df = DeepFakeDetector()       # prithivMLmods/Deep-Fake-Detector-v2-Model
        self.model_clip = CLIPDetector()          # CLIP semantic

    def predict(self, image_path):
        # ── ML Model Scores ──
        try:
            score_ai = max(0.0, min(self.model_ai.predict(image_path), 1.0))
        except:
            score_ai = 0.0

        try:
            score_df = max(0.0, min(self.model_df.predict(image_path), 1.0))
        except:
            score_df = 0.0

        try:
            score_clip = max(0.0, min(self.model_clip.predict(image_path), 1.0))
        except:
            score_clip = 0.0

        # ── Heuristic Scores ──
        try:
            score_quality = max(0.0, min(image_quality_score(image_path), 1.0))
        except:
            score_quality = 0.0

        try:
            score_freq = max(0.0, min(frequency_score(image_path), 1.0))
        except:
            score_freq = 0.0

        try:
            score_noise = max(0.0, min(noise_score(image_path), 1.0))
        except:
            score_noise = 0.0

        # ── Combine ML models: take the MAX of the two dedicated detectors ──
        # If EITHER model catches it as fake, we trust the higher score
        ml_score = max(score_ai, score_df)

        # 🔥 Final weighted score
        final_score = (
            0.45 * ml_score +       # Best ML detector result
            0.20 * score_clip +      # CLIP semantic
            0.15 * score_quality +   # Sharpness heuristic
            0.10 * score_freq +      # Frequency domain
            0.10 * score_noise       # Noise pattern
        )

        return max(0.0, min(final_score, 1.0))
