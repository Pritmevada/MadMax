"""
MadMax — Fake Profile Detection Pipeline (End-to-End)

Usage (run from inside the genai folder):
    python profile_pipeline.py              # analyzes FOLDER_PATH below
    python profile_pipeline.py data2        # analyzes data2 folder
"""

FOLDER_PATH = "data"    # ← Change to "data2", "data3", etc.

import os
import sys
from image_pipeline import analyze_images
from text_pipeline import analyze_text


def classify(score):
    """Human-readable verdict from a 0-1 score."""
    if score >= 0.70:
        return "🔴 FAKE (High Confidence)"
    elif score >= 0.50:
        return "🟠 SUSPICIOUS (Medium Confidence)"
    elif score >= 0.30:
        return "🟡 UNCERTAIN (Low Confidence)"
    else:
        return "🟢 LIKELY GENUINE"


def run_pipeline(folder_path="data"):
    print("\n" + "━" * 55)
    print("   🔍 MadMax — Fake Profile Detection Pipeline")
    print("━" * 55)
    print(f"   Analyzing folder: {os.path.abspath(folder_path)}")
    print("━" * 55)

    # ── IMAGE ANALYSIS ──
    print("\n📷 IMAGE ANALYSIS")
    print("-" * 45)
    try:
        image_score = analyze_images(folder_path)
    except Exception as e:
        print(f"   ⚠️  Image analysis failed: {e}")
        image_score = 0.0

    image_confidence = image_score * 100
    image_label = "🔴 FAKE" if image_score >= 0.5 else "🟢 REAL"
    print(f"\n   Overall Image Score: {image_score:.4f} → {image_label}  (Confidence: {image_confidence:.2f}%)")

    # ── TEXT ANALYSIS ──
    print("\n📝 TEXT ANALYSIS")
    print("-" * 45)
    try:
        text_results = analyze_text(folder_path)
        text_score = text_results["final_score"]
    except Exception as e:
        print(f"   ⚠️  Text analysis failed: {e}")
        text_score = 0.0
        text_results = {}

    text_confidence = text_score * 100
    text_label = "🔴 SUSPICIOUS" if text_score >= 0.5 else "🟢 NORMAL"

    print(f"\n   ── Text Score Breakdown ──")
    print(f"   Spam:               {text_results.get('spam', 0) * 100:.2f}%")
    print(f"   Semantic Coherence:  {text_results.get('coherence', 0) * 100:.2f}%")
    print(f"   Engagement Bait:     {text_results.get('engagement_bait', 0) * 100:.2f}%")
    print(f"   Bio Suspicion:       {text_results.get('bio_suspicion', 0) * 100:.2f}%")
    print(f"   Repetition:          {text_results.get('repetition', 0) * 100:.2f}%")
    print(f"\n   Overall Text Score: {text_score:.4f} → {text_label}  (Confidence: {text_confidence:.2f}%)")

    # ── FINAL COMBINED SCORE ──
    final_score = (0.45 * image_score) + (0.55 * text_score)

    # Boost if BOTH signals are high
    if image_score >= 0.5 and text_score >= 0.5:
        final_score = min(final_score * 1.15, 1.0)

    overall_confidence = final_score * 100
    verdict = classify(final_score)

    print("\n" + "═" * 55)
    print("              📊 FINAL RESULTS")
    print("═" * 55)
    print(f"   Image Score:       {image_score:.4f}  ({image_confidence:.2f}%)")
    print(f"   Text Score:        {text_score:.4f}  ({text_confidence:.2f}%)")
    print(f"   ─────────────────────────────────")
    print(f"   COMBINED SCORE:    {final_score:.4f}")
    print(f"   OVERALL CONFIDENCE: {overall_confidence:.2f}%")
    print(f"   VERDICT:           {verdict}")
    print("═" * 55 + "\n")

    return {
        "image_score": image_score,
        "text_score": text_score,
        "final_score": final_score,
        "overall_confidence": overall_confidence,
        "verdict": verdict,
    }


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else FOLDER_PATH
    run_pipeline(folder)
