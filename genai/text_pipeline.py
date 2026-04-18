import os
import re
from text_detector import TextAnalyzer


# ──────────────────────────────────────────────
# 1. SPAM DETECTION — keyword + pattern matching
# ──────────────────────────────────────────────
SPAM_PHRASES = [
    "dm me", "link in bio", "earn money", "crypto", "follow me",
    "passive income", "financial freedom", "message me",
    "opportunity", "investment", "grow your wealth",
    "make money", "get rich", "free money", "work from home",
    "limited time", "act now", "don't miss out", "join now",
    "click the link", "swipe up", "check bio", "inbox me",
    "100% guaranteed", "no investment", "easy money",
    "trading", "forex", "nft", "bitcoin", "hustle",
    "grind", "boss life", "millionaire mindset",
]

EMOJI_SPAM = re.compile(
    r"[🚀💰🔥💡💸💪📈👇💵🤑💎⚡️✨🏆🥇💯]"
)

def spam_score(text_list):
    if not text_list:
        return 0.0

    total_hits = 0
    total_emoji_hits = 0

    for text in text_list:
        lower = text.lower()
        for phrase in SPAM_PHRASES:
            if phrase in lower:
                total_hits += 1
        total_emoji_hits += len(EMOJI_SPAM.findall(text))

    keyword_density = min(total_hits / (len(text_list) * 2), 1.0)
    emoji_density = min(total_emoji_hits / (len(text_list) * 3), 1.0)

    return 0.7 * keyword_density + 0.3 * emoji_density


# ──────────────────────────────────────────────
# 2. REPETITION
# ──────────────────────────────────────────────
def repetition_score(captions):
    if len(captions) <= 1:
        return 0.0
    unique = set(captions)
    return 1.0 - (len(unique) / len(captions))


# ──────────────────────────────────────────────
# 3. SEMANTIC SIMILARITY — bio ↔ captions
# ──────────────────────────────────────────────
def semantic_coherence_score(analyzer, bio, captions):
    if not captions or not bio:
        return 0.0

    similarities = []
    for caption in captions:
        sim = analyzer.similarity(bio, caption)
        similarities.append(sim)

    avg_sim = sum(similarities) / len(similarities)

    cross_sims = []
    for i in range(len(captions)):
        for j in range(i + 1, min(i + 5, len(captions))):
            cross_sims.append(analyzer.similarity(captions[i], captions[j]))

    avg_cross = sum(cross_sims) / len(cross_sims) if cross_sims else 0.0

    bio_suspicion = min(avg_sim / 0.6, 1.0)
    cross_suspicion = min(avg_cross / 0.7, 1.0)

    return 0.6 * bio_suspicion + 0.4 * cross_suspicion


# ──────────────────────────────────────────────
# 4. ENGAGEMENT BAIT
# ──────────────────────────────────────────────
def engagement_bait_score(captions):
    if not captions:
        return 0.0

    cta_patterns = [
        r"dm\s+me", r"message\s+me", r"follow\s+me", r"link\s+in\s+bio",
        r"comment\s+below", r"tag\s+a\s+friend", r"share\s+this",
        r"click\s+(the\s+)?link", r"swipe\s+up", r"check\s+(my\s+)?bio",
        r"inbox\s+me", r"join\s+(now|today|me)",
    ]

    short_count = 0
    cta_count = 0

    for cap in captions:
        words = cap.split()
        if len(words) < 15:
            short_count += 1

        lower = cap.lower()
        for pattern in cta_patterns:
            if re.search(pattern, lower):
                cta_count += 1
                break

    short_ratio = short_count / len(captions)
    cta_ratio = cta_count / len(captions)

    return 0.4 * short_ratio + 0.6 * cta_ratio


# ──────────────────────────────────────────────
# 5. BIO RED FLAGS
# ──────────────────────────────────────────────
def bio_suspicion_score(bio):
    if not bio:
        return 0.0

    score = 0.0
    lower = bio.lower()

    emoji_count = len(EMOJI_SPAM.findall(bio))
    if emoji_count >= 3:
        score += 0.3

    if bio.count("|") >= 2:
        score += 0.2

    money_keywords = ["financial freedom", "passive income", "crypto", "trading",
                      "investment", "earn", "money", "wealth", "millionaire",
                      "forex", "nft", "bitcoin"]
    hits = sum(1 for kw in money_keywords if kw in lower)
    if hits >= 2:
        score += 0.3

    if any(cta in lower for cta in ["dm me", "message me", "link in bio", "click", "inbox"]):
        score += 0.2

    return min(score, 1.0)


# ══════════════════════════════════════════════
# MAIN TEXT ANALYSIS
# ══════════════════════════════════════════════
def analyze_text(folder_path):
    analyzer = TextAnalyzer()

    # Auto-detect bio and caption files
    bio = ""
    captions = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".txt"):
            continue
        filepath = os.path.join(folder_path, filename)
        lower = filename.lower()

        if lower.startswith("bio"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    bio = content

        elif lower.startswith("caption"):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                captions.extend(lines)

    if not bio:
        print("   ⚠️  No bio file found (bio.txt, bio2.txt, etc.)")
        return {"final_score": 0.0}

    if not captions:
        print("   ⚠️  No caption files found (captions.txt, caption1.txt, etc.)")
        return {"final_score": 0.0}

    all_text = [bio] + captions

    # ── Per-caption confidence ──
    print("\n   📄 BIO ANALYSIS:")
    bio_sus = bio_suspicion_score(bio)
    bio_label = "🔴 SUSPICIOUS" if bio_sus >= 0.5 else "🟢 NORMAL"
    print(f"      Bio → {bio_label}  (Confidence: {bio_sus * 100:.2f}%)")

    print("\n   💬 CAPTION ANALYSIS:")
    caption_spam_scores = []
    for i, caption in enumerate(captions, 1):
        cap_spam = spam_score([caption])
        cap_engagement = engagement_bait_score([caption])
        cap_score = 0.6 * cap_spam + 0.4 * cap_engagement
        caption_spam_scores.append(cap_score)

        label = "🔴 SPAM" if cap_score >= 0.4 else "🟡 BORDERLINE" if cap_score >= 0.2 else "🟢 CLEAN"
        short_caption = caption[:50] + "..." if len(caption) > 50 else caption
        print(f"      Caption {i:<3} → {label}  (Confidence: {cap_score * 100:.2f}%)  \"{short_caption}\"")

    # ── Aggregate scores ──
    spam = spam_score(all_text)
    rep = repetition_score(captions)
    coherence = semantic_coherence_score(analyzer, bio, captions)
    engagement = engagement_bait_score(captions)

    final_text_score = (
        0.30 * spam +
        0.25 * coherence +
        0.20 * engagement +
        0.15 * bio_sus +
        0.10 * rep
    )

    final_text_score = max(0.0, min(final_text_score, 1.0))

    results = {
        "spam": spam,
        "coherence": coherence,
        "engagement_bait": engagement,
        "bio_suspicion": bio_sus,
        "repetition": rep,
        "final_score": final_text_score,
    }

    return results


# ══════════════════════════════════════════════
# TEST DIRECTLY
# ══════════════════════════════════════════════
if __name__ == "__main__":
    folder = "data"
    results = analyze_text(folder)

    print("\n" + "=" * 45)
    print("         TEXT ANALYSIS BREAKDOWN")
    print("=" * 45)
    print(f"  Spam Score:            {results.get('spam', 0):.4f}  ({results.get('spam', 0) * 100:.2f}%)")
    print(f"  Semantic Coherence:    {results.get('coherence', 0):.4f}  ({results.get('coherence', 0) * 100:.2f}%)")
    print(f"  Engagement Bait:       {results.get('engagement_bait', 0):.4f}  ({results.get('engagement_bait', 0) * 100:.2f}%)")
    print(f"  Bio Suspicion:         {results.get('bio_suspicion', 0):.4f}  ({results.get('bio_suspicion', 0) * 100:.2f}%)")
    print(f"  Repetition:            {results.get('repetition', 0):.4f}  ({results.get('repetition', 0) * 100:.2f}%)")
    print("-" * 45)
    print(f"  FINAL TEXT SCORE:      {results['final_score']:.4f}  (Confidence: {results['final_score'] * 100:.2f}%)")
    print("=" * 45)
