"""
train_model_real.py
Trains a TF-IDF + Logistic Regression sentiment classifier on course reviews.

Sentiment mapping from 1-5 star ratings:
    1-2 stars -> negative
    3 stars   -> neutral
    4-5 stars -> positive

Saves:
    tfidf_vectorizer.pkl
    course_sentiment_model.pkl
    metrics.json
"""

import json
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "course_reviews.csv"


def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Review"]).copy()
    df["Review"] = df["Review"].astype(str)
    df["sentiment"] = df["Label"].apply(label_sentiment)

    X = df["Review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=5,
    )
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)

    report = classification_report(y_test, y_pred, output_dict=True, digits=3)
    print(classification_report(y_test, y_pred, digits=3))

    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    print("Confusion matrix (rows=actual, cols=predicted):", labels)
    print(cm)

    metrics = {
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("course_sentiment_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("\nSaved: tfidf_vectorizer.pkl, course_sentiment_model.pkl, metrics.json")


if __name__ == "__main__":
    main()
