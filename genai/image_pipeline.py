import os
from advanced_image_detector import AdvancedDetector

def analyze_images(folder_path):
    detector = AdvancedDetector()
    scores = []

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, file)

            try:
                score = detector.predict(path)
                confidence = score * 100
                label = "🔴 FAKE" if score >= 0.5 else "🟢 REAL"
                print(f"   {file:<25} → {label}  (Confidence: {confidence:.2f}%)")
                scores.append(score)
            except Exception as e:
                print(f"   {file:<25} → ⚠️  Error: {e}")

    if not scores:
        return 0.0

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    # Weighted logic
    final_score = (0.6 * avg_score) + (0.4 * max_score)
    return final_score


# 👉 RUN TEST DIRECTLY FROM THIS FILE
if __name__ == "__main__":
    folder = "data"
    final_score = analyze_images(folder)

    print("\n======================")
    print(f"FINAL IMAGE AI SCORE: {final_score:.4f} (Confidence: {final_score * 100:.2f}%)")
    print("======================")
