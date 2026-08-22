"""
Aspect-Based Sentiment Analysis
--------------------------------
Instead of giving ONE sentiment label for a whole review, this module
breaks a review into clauses and reports sentiment PER ASPECT
(fabric/quality, fit/sizing, price/value, delivery/shipping, color/appearance).

Example:
    "Fabric is great but delivery was very late"
    ->  {"Fabric/Quality": "POSITIVE", "Delivery/Shipping": "NEGATIVE"}

This reuses the existing TF-IDF + Logistic Regression model — no new
model or extra data needed. It just applies it more cleverly.
"""

import re

# ---------- Aspect keyword dictionary ----------
# Add/edit keywords here to tune which words trigger which aspect.
ASPECT_KEYWORDS = {
    "Fabric/Quality": [
        "fabric", "material", "quality", "stitching", "seam", "cloth",
        "fiber", "texture", "thread", "knit", "weave"
    ],
    "Fit/Sizing": [
        "size", "sizing", "fit", "fits", "tight", "loose", "small",
        "large", "length", "waist", "true to size", "runs big", "runs small"
    ],
    "Price/Value": [
        "price", "expensive", "cheap", "worth", "value", "cost",
        "money", "overpriced", "affordable", "pricey"
    ],
    "Delivery/Shipping": [
        "delivery", "shipping", "arrived", "late", "shipment",
        "package", "packaging", "box", "courier", "delay", "delayed"
    ],
    "Color/Appearance": [
        "color", "colour", "design", "look", "looks", "style",
        "pattern", "print", "appearance", "shade", "faded"
    ],
}

# Words/punctuation used to split a review into smaller clauses,
# so different opinions in the same sentence don't get mixed together.
_SPLIT_PATTERN = re.compile(
    r",|;|\.|(?:\s+but\s+)|(?:\s+however\s+)|(?:\s+although\s+)|(?:\s+while\s+)|(?:\s+though\s+)",
    flags=re.IGNORECASE,
)

# The base clothing-review dataset barely discusses delivery/shipping (it's
# about fabric, fit, and style) — so the ML model has very weak signal for
# words like "delivery", "delayed", "damaged". Rather than trust a low-signal
# ML guess, we override with simple, explainable keyword rules for this one
# aspect. This is a documented design choice, not silently hidden behavior.
DELIVERY_OVERRIDES = {
    "NEGATIVE": [
        "late", "delayed", "delay", "slow", "damaged", "broken box",
        "lost package", "never arrived", "took forever", "long time to arrive",
        "wrong item", "missing",
    ],
    "POSITIVE": [
        "fast", "quick", "on time", "arrived early", "prompt",
        "well packaged", "great packaging", "arrived quickly",
    ],
}


def _apply_delivery_override(clause: str):
    """Returns 'NEGATIVE'/'POSITIVE' if a strong delivery keyword is found, else None."""
    clause_lower = clause.lower()
    for word in DELIVERY_OVERRIDES["NEGATIVE"]:
        if word in clause_lower:
            return "NEGATIVE"
    for word in DELIVERY_OVERRIDES["POSITIVE"]:
        if word in clause_lower:
            return "POSITIVE"
    return None


def split_into_clauses(review_text: str):
    """Split a review into smaller clauses on conjunctions/punctuation."""
    clauses = _SPLIT_PATTERN.split(review_text)
    return [c.strip() for c in clauses if c and c.strip()]


def match_aspects(clause: str):
    """Return list of aspect names whose keywords appear in this clause."""
    clause_lower = clause.lower()
    matched = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in clause_lower for kw in keywords):
            matched.append(aspect)
    return matched


def _distilbert_sentiment(clause: str, distilbert_pipe):
    """Runs DistilBERT on a clause and maps to POSITIVE/NEGATIVE/NEUTRAL,
    using the same threshold logic as the Live Workspace tab."""
    db_raw = distilbert_pipe(clause)
    res_list = db_raw[0] if isinstance(db_raw, list) and isinstance(db_raw[0], list) else db_raw
    db_scores = {item['label'].upper(): item['score'] for item in res_list if isinstance(item, dict)}
    pos_score = db_scores.get('POSITIVE', 0.0)
    neg_score = db_scores.get('NEGATIVE', 0.0)
    if pos_score > neg_score and pos_score > 0.5:
        return "POSITIVE"
    elif neg_score > pos_score and neg_score > 0.5:
        return "NEGATIVE"
    return "NEUTRAL"


def analyze_review_aspects(review_text: str, model, vectorizer, engine="tfidf", distilbert_pipe=None):
    """
    Returns a dict: {aspect_name: (sentiment_label, source_clause)}
    for every aspect mentioned in the review.

    engine: "tfidf" (default), "distilbert", or "ensemble".
    In "ensemble" mode, the returned sentiment is the DistilBERT result
    (matching how the Live Workspace tab resolves ensemble mode), but this
    function can be called twice (once per engine) if you want to display
    both side by side.
    """
    results = {}
    clauses = split_into_clauses(review_text)

    for clause in clauses:
        aspects_here = match_aspects(clause)
        if not aspects_here:
            continue

        if engine == "distilbert" and distilbert_pipe is not None:
            base_sentiment = _distilbert_sentiment(clause, distilbert_pipe)
        else:
            vec = vectorizer.transform([clause])
            base_sentiment = "NEUTRAL" if vec.nnz == 0 else model.predict(vec)[0]

        for aspect in aspects_here:
            if aspect == "Delivery/Shipping":
                override = _apply_delivery_override(clause)
                sentiment = override if override else base_sentiment
            else:
                sentiment = base_sentiment
            # If the same aspect appears twice, keep the first (usually clearer) mention
            results.setdefault(aspect, (sentiment, clause))

    return results


def analyze_batch(reviews, model, vectorizer):
    """
    Runs aspect analysis across many reviews and aggregates counts.
    Returns a list of rows: {"Aspect": ..., "Sentiment": ..., "Count": ...}
    ready to be turned into a bar chart or summary table.
    """
    from collections import Counter
    counter = Counter()

    for review in reviews:
        if not isinstance(review, str) or not review.strip():
            continue
        aspects = analyze_review_aspects(review, model, vectorizer)
        for aspect, (sentiment, _clause) in aspects.items():
            counter[(aspect, sentiment)] += 1

    rows = [
        {"Aspect": aspect, "Sentiment": sentiment, "Count": count}
        for (aspect, sentiment), count in counter.items()
    ]
    return rows


if __name__ == "__main__":
    # Quick manual test (requires the trained model files in this folder)
    import joblib

    model = joblib.load("clothing_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    test_reviews = [
        "Fabric is great but delivery was very late",
        "The price is too high for such cheap material",
        "Perfect fit, true to size, and the color is beautiful",
        "Loved the design but it arrived in a damaged box",
    ]

    for r in test_reviews:
        print(f"\nReview: {r}")
        result = analyze_review_aspects(r, model, vectorizer)
        for aspect, (sentiment, clause) in result.items():
            print(f"  {aspect}: {sentiment}   (from: \"{clause}\")")
