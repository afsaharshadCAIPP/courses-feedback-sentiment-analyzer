"""
aspect_analyzer.py
Simple keyword-based aspect extraction for course reviews.

Identifies which aspect of a course a review is talking about
(content, instructor, pacing, assignments, structure) so that
sentiment can optionally be broken down per-aspect instead of
just per-review.
"""

import re
from collections import defaultdict

ASPECT_KEYWORDS = {
    "content": ["content", "material", "topics", "curriculum", "syllabus"],
    "instructor": ["instructor", "teacher", "professor", "lecturer", "explanation"],
    "pacing": ["pace", "pacing", "speed", "fast", "slow", "rushed"],
    "assignments": ["assignment", "homework", "project", "exercise", "quiz"],
    "structure": ["structure", "organized", "organised", "structured", "flow"],
}


def extract_aspects(text: str):
    """Return a list of aspects mentioned in the given review text."""
    text_lower = text.lower()
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords):
            found.append(aspect)
    return found


def analyze_reviews(reviews, sentiments):
    """
    Given parallel lists of review texts and their predicted sentiments,
    return a dict: aspect -> {sentiment: count}
    """
    aspect_sentiment_counts = defaultdict(lambda: defaultdict(int))

    for text, sentiment in zip(reviews, sentiments):
        aspects = extract_aspects(text)
        for aspect in aspects:
            aspect_sentiment_counts[aspect][sentiment] += 1

    return {aspect: dict(counts) for aspect, counts in aspect_sentiment_counts.items()}


if __name__ == "__main__":
    sample_reviews = [
        "The instructor explained everything clearly and the pacing was perfect.",
        "Content was outdated and the assignments were too easy.",
        "Great structure, but the pacing was too fast for beginners.",
    ]
    sample_sentiments = ["positive", "negative", "neutral"]

    result = analyze_reviews(sample_reviews, sample_sentiments)
    for aspect, counts in result.items():
        print(f"{aspect}: {counts}")
