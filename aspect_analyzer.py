# ============================================================
# ASPECT-BASED SENTIMENT ANALYZER
# Coursera Course Feedback Sentiment Analysis
#
# Uses the ALREADY TRAINED:
# 1. Multilingual DistilBERT model
# 2. TF-IDF + Logistic Regression model
#
# The analyzer:
# 1. Accepts a course review
# 2. Splits the review into sentences
# 3. Detects predefined course-feedback aspects
# 4. Extracts the relevant sentence(s)
# 5. Predicts sentiment using DistilBERT
# 6. Optionally compares with TF-IDF + Logistic Regression
# 7. Produces aspect-level sentiment results
#
# Aspect categories:
#   - Course Content
#   - Instructor
#   - Assignments
#   - Difficulty
#   - Learning Experience
#   - Course Structure
#   - Platform
#   - Certificates
#   - Duration
#   - Value
#
# ============================================================


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (AutoTokenizer,AutoModelForSequenceClassification)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DISTILBERT_MODEL_DIR = "./coursera_multilingual_distilbert"

TFIDF_MODEL_FILE = "./coursera_tfidf_logistic_model.pkl"

TFIDF_VECTORIZER_FILE = "./coursera_tfidf_vectorizer.pkl"

MAX_LENGTH = 128

LABEL_NAMES = [
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE"
]


# ============================================================
# 2. FORCE CPU
# ============================================================

DEVICE = torch.device("cpu")

print("\n" + "=" * 60)
print("DEVICE CONFIGURATION")
print("=" * 60)
print("Running on:", DEVICE)
print("CUDA will NOT be used.")


# ============================================================
# 3. ASPECT KEYWORDS
# ============================================================

ASPECT_KEYWORDS = {

    "Course Content": [
        "content",
        "course content",
        "material",
        "materials",
        "course material",
        "learning material",
        "topics",
        "topic",
        "lessons",
        "lesson",
        "curriculum",
        "concepts",
        "concept",
        "theory",
        "information"
    ],

    "Instructor": [
        "instructor",
        "teacher",
        "professor",
        "lecturer",
        "mentor",
        "teaching",
        "teach",
        "taught",
        "explained",
        "explanation",
        "lecture",
        "lectures"
    ],

    "Assignments": [
        "assignment",
        "assignments",
        "homework",
        "exercise",
        "exercises",
        "quiz",
        "quizzes",
        "test",
        "tests",
        "project",
        "projects",
        "practice",
        "practical"
    ],

    "Difficulty": [
        "difficult",
        "difficulty",
        "easy",
        "easier",
        "hard",
        "challenging",
        "challenge",
        "complex",
        "complicated",
        "simple",
        "beginner",
        "advanced"
    ],

    "Learning Experience": [
        "learn",
        "learned",
        "learning",
        "experience",
        "understand",
        "understanding",
        "helpful",
        "useful",
        "skill",
        "skills",
        "improved",
        "improve"
    ],

    "Course Structure": [
        "structure",
        "structured",
        "organized",
        "organised",
        "organization",
        "organisation",
        "sequence",
        "order",
        "module",
        "modules",
        "section",
        "sections"
    ],

    "Platform": [
        "platform",
        "website",
        "site",
        "interface",
        "video",
        "videos",
        "audio",
        "subtitles",
        "subtitle",
        "captions",
        "caption",
        "app",
        "application",
        "coursera"
    ],

    "Certificates": [
        "certificate",
        "certification",
        "certified",
        "credential"
    ],

    "Duration": [
        "duration",
        "length",
        "time",
        "hours",
        "hour",
        "week",
        "weeks",
        "short",
        "long"
    ],

    "Value": [
        "value",
        "worth",
        "price",
        "cost",
        "money",
        "paid",
        "investment",
        "recommend",
        "recommended"
    ]
}


# ============================================================
# 4. LOAD DISTILBERT MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MULTILINGUAL DISTILBERT")
print("=" * 60)

# Prefer a local copy of the model if present, otherwise load straight from
# the Hugging Face Hub repo (the fine-tuned model is hosted there since the
# weights file is too large for a normal GitHub push).
DISTILBERT_SOURCE = DISTILBERT_MODEL_DIR if os.path.exists(DISTILBERT_MODEL_DIR) else "Afsah-2027/coursera-multilingual-distilbert"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    DISTILBERT_SOURCE
)

print("Loading model...")

distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    DISTILBERT_SOURCE
)

distilbert_model.to(DEVICE)

distilbert_model.eval()

print("DistilBERT loaded successfully.")
print("Model device:", DEVICE)


# ============================================================
# 5. LOAD TF-IDF + LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LOADING TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

tfidf_model = None
tfidf_vectorizer = None

if (
    os.path.exists(TFIDF_MODEL_FILE)
    and
    os.path.exists(TFIDF_VECTORIZER_FILE)
):

    tfidf_model = joblib.load(
        TFIDF_MODEL_FILE
    )

    tfidf_vectorizer = joblib.load(
        TFIDF_VECTORIZER_FILE
    )

    print(
        "TF-IDF + Logistic Regression loaded successfully."
    )

else:

    print(
        "WARNING: TF-IDF model/vectorizer not found."
    )

    print(
        "The analyzer will use DistilBERT only."
    )


# ============================================================
# 6. CLEAN TEXT
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# 7. SPLIT INTO SENTENCES
# ============================================================

def split_into_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?。！？])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# 8. DETECT ASPECTS
# ============================================================

def find_aspects(sentence):

    sentence_lower = sentence.lower()

    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword.lower())
                + r"\b"
            )

            if re.search(
                pattern,
                sentence_lower
            ):

                detected_aspects.append(
                    aspect
                )

                break

    return detected_aspects


# ============================================================
# 9. EXTRACT ASPECT-RELATED SENTENCES
# ============================================================

def extract_aspect_sentences(text):

    sentences = split_into_sentences(text)

    aspect_sentences = {
        aspect: []
        for aspect in ASPECT_KEYWORDS
    }

    for sentence in sentences:

        aspects = find_aspects(sentence)

        for aspect in aspects:

            aspect_sentences[aspect].append(
                sentence
            )

    return aspect_sentences


# ============================================================
# 10. DISTILBERT PREDICTION
# ============================================================

def predict_distilbert(text):

    text = clean_text(text)

    if not text:

        return {
            "sentiment": "UNKNOWN",
            "confidence": 0.0,
            "probabilities": {}
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = distilbert_model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        prediction
    ].item()

    probability_values = (
        probabilities[0]
        .cpu()
        .numpy()
    )

    return {

        "sentiment":
            LABEL_NAMES[prediction],

        "confidence":
            float(confidence),

        "probabilities": {

            "NEGATIVE":
                float(probability_values[0]),

            "NEUTRAL":
                float(probability_values[1]),

            "POSITIVE":
                float(probability_values[2])
        }
    }


# ============================================================
# 11. TF-IDF PREDICTION
# ============================================================

def predict_tfidf(text):

    if (
        tfidf_model is None
        or
        tfidf_vectorizer is None
    ):

        return {
            "sentiment": "UNAVAILABLE",
            "confidence": 0.0
        }

    text = clean_text(text)

    if not text:

        return {
            "sentiment": "UNKNOWN",
            "confidence": 0.0
        }

    vector = tfidf_vectorizer.transform(
        [text]
    )

    prediction = tfidf_model.predict(
        vector
    )[0]

    if hasattr(
        tfidf_model,
        "predict_proba"
    ):

        probabilities = (
            tfidf_model
            .predict_proba(vector)[0]
        )

        confidence = float(
            np.max(probabilities)
        )

    else:

        confidence = 0.0

    return {

        "sentiment":
            str(prediction).upper(),

        "confidence":
            confidence
    }


# ============================================================
# 12. ANALYZE ONE REVIEW
# ============================================================

def analyze_review(
    review,
    use_tfidf=True
):

    review = clean_text(review)

    if not review:

        return {
            "review": "",
            "overall_sentiment": "UNKNOWN",
            "aspects": {}
        }

    aspect_sentences = (
        extract_aspect_sentences(
            review
        )
    )

    results = {}

    for aspect, sentences in aspect_sentences.items():

        if not sentences:
            continue

        unique_sentences = list(
            dict.fromkeys(sentences)
        )

        aspect_text = " ".join(
            unique_sentences
        )

        distilbert_result = (
            predict_distilbert(
                aspect_text
            )
        )

        if use_tfidf:

            tfidf_result = (
                predict_tfidf(
                    aspect_text
                )
            )

        else:

            tfidf_result = {
                "sentiment": "DISABLED",
                "confidence": 0.0
            }

        results[aspect] = {

            "sentiment":
                distilbert_result[
                    "sentiment"
                ],

            "confidence":
                round(
                    distilbert_result[
                        "confidence"
                    ],
                    4
                ),

            "distilbert_probabilities":
                {
                    key:
                    round(
                        value,
                        4
                    )
                    for key, value
                    in distilbert_result[
                        "probabilities"
                    ].items()
                },

            "tfidf_sentiment":
                tfidf_result[
                    "sentiment"
                ],

            "tfidf_confidence":
                round(
                    tfidf_result[
                        "confidence"
                    ],
                    4
                ),

            "evidence":
                unique_sentences
        }

    # ========================================================
    # OVERALL SENTIMENT
    # ========================================================

    if results:

        sentiment_values = [
            data["sentiment"]
            for data in results.values()
            if data["sentiment"]
            in LABEL_NAMES
        ]

        if sentiment_values:

            sentiment_scores = {
                "NEGATIVE": 0,
                "NEUTRAL": 1,
                "POSITIVE": 2
            }

            numerical_values = [
                sentiment_scores[sentiment]
                for sentiment in sentiment_values
            ]

            average_score = (
                sum(numerical_values)
                /
                len(numerical_values)
            )

            if average_score < 0.75:

                overall_sentiment = "NEGATIVE"

            elif average_score < 1.5:

                overall_sentiment = "NEUTRAL"

            else:

                overall_sentiment = "POSITIVE"

        else:

            overall_sentiment = "UNKNOWN"

    else:

        prediction = predict_distilbert(
            review
        )

        overall_sentiment = (
            prediction["sentiment"]
        )

    return {

        "review":
            review,

        "overall_sentiment":
            overall_sentiment,

        "aspects":
            results
    }


# ============================================================
# 13. PRINT RESULT
# ============================================================

def print_analysis(result):

    print("\n")
    print("=" * 70)
    print("ASPECT-BASED SENTIMENT ANALYSIS")
    print("=" * 70)

    print("\nREVIEW:")
    print(result["review"])

    print(
        "\nOVERALL SENTIMENT:",
        result["overall_sentiment"]
    )

    print("\nASPECT RESULTS:")
    print("-" * 70)

    if not result["aspects"]:

        print(
            "No predefined aspects detected."
        )

        return

    for aspect, data in result["aspects"].items():

        print(
            f"\nAspect: {aspect}"
        )

        print(
            f"Sentiment: {data['sentiment']}"
        )

        print(
            f"DistilBERT Confidence: "
            f"{data['confidence']:.4f}"
        )

        print(
            f"TF-IDF Sentiment: "
            f"{data['tfidf_sentiment']}"
        )

        print(
            f"TF-IDF Confidence: "
            f"{data['tfidf_confidence']:.4f}"
        )

        print("Evidence:")

        for sentence in data["evidence"]:

            print(
                f"  - {sentence}"
            )


# ============================================================
# 14. RESULT TO DATAFRAME
# ============================================================

def result_to_dataframe(result):

    rows = []

    for aspect, data in result["aspects"].items():

        rows.append({

            "Aspect":
                aspect,

            "Sentiment":
                data["sentiment"],

            "DistilBERT Confidence":
                data["confidence"],

            "TF-IDF Sentiment":
                data["tfidf_sentiment"],

            "TF-IDF Confidence":
                data["tfidf_confidence"],

            "Evidence":
                " ".join(
                    data["evidence"]
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# 15. ANALYZE MULTIPLE REVIEWS
# ============================================================

def analyze_multiple_reviews(reviews):

    all_rows = []

    for review in reviews:

        result = analyze_review(
            review
        )

        for aspect, data in result["aspects"].items():

            all_rows.append({

                "Review":
                    review,

                "Aspect":
                    aspect,

                "Sentiment":
                    data["sentiment"],

                "Confidence":
                    data["confidence"],

                "TF-IDF Sentiment":
                    data["tfidf_sentiment"],

                "TF-IDF Confidence":
                    data["tfidf_confidence"],

                "Evidence":
                    " ".join(
                        data["evidence"]
                    )
            })

    return pd.DataFrame(
        all_rows
    )


# ============================================================
# 16. ASPECT SUMMARY
# ============================================================

def create_aspect_summary(
    analysis_dataframe
):

    if analysis_dataframe.empty:

        return pd.DataFrame()

    summary_rows = []

    for aspect, group in analysis_dataframe.groupby(
        "Aspect"
    ):

        total = len(group)

        positive_count = (
            group["Sentiment"]
            .eq("POSITIVE")
            .sum()
        )

        neutral_count = (
            group["Sentiment"]
            .eq("NEUTRAL")
            .sum()
        )

        negative_count = (
            group["Sentiment"]
            .eq("NEGATIVE")
            .sum()
        )

        summary_rows.append({

            "Aspect":
                aspect,

            "Total Mentions":
                total,

            "Positive":
                positive_count,

            "Neutral":
                neutral_count,

            "Negative":
                negative_count,

            "Positive %":
                round(
                    positive_count
                    / total
                    * 100,
                    2
                ),

            "Neutral %":
                round(
                    neutral_count
                    / total
                    * 100,
                    2
                ),

            "Negative %":
                round(
                    negative_count
                    / total
                    * 100,
                    2
                )
        })

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# 17. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("COURSE FEEDBACK ASPECT ANALYZER")
    print("=" * 70)

    print(
        "\nModels loaded successfully."
    )

    # --------------------------------------------------------
    # Test reviews
    # --------------------------------------------------------

    test_reviews = [

        (
            "The instructor explained the concepts "
            "very clearly. The course content was "
            "excellent but the assignments were "
            "quite difficult."
        ),

        (
            "The videos were good and the material "
            "was useful. However, the course was "
            "poorly organized."
        ),

        (
            "This course was easy to follow and "
            "I learned a lot. The instructor was "
            "excellent."
        ),

        (
            "The assignments were too hard and "
            "the videos were too long. The "
            "certificate was useful."
        )
    ]

    # --------------------------------------------------------
    # Analyze test reviews
    # --------------------------------------------------------

    for review in test_reviews:

        result = analyze_review(
            review
        )

        print_analysis(
            result
        )

        dataframe = result_to_dataframe(
            result
        )

        print("\nDataFrame:")

        if not dataframe.empty:

            print(
                dataframe.to_string(
                    index=False
                )
            )

        else:

            print(
                "No aspects detected."
            )

    # ========================================================
    # INTERACTIVE MODE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)

    print(
        "\nEnter a course review to analyze."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        try:

            review = input(
                "\nEnter review: "
            )

        except KeyboardInterrupt:

            print(
                "\n\nProgram stopped."
            )

            break

        if review.lower().strip() == "exit":

            print(
                "\nAspect analyzer stopped."
            )

            break

        if not review.strip():

            print(
                "Please enter a review."
            )

            continue

        result = analyze_review(
            review
        )

        print_analysis(
            result
        )

        print("\nJSON result:")

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )