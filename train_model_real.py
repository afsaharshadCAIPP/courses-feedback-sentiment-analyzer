"""
Trains the TF-IDF + Logistic Regression model on the REAL
"Women's Clothing E-Commerce Reviews" dataset (23,486 rows).
Sentiment label is derived from the star Rating column:
    Rating 1-2 -> NEGATIVE
    Rating 3   -> NEUTRAL
    Rating 4-5 -> POSITIVE
Saves REAL evaluation metrics to metrics.json (no fake/hardcoded numbers).
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------- Step 1: Load real dataset ----------
df = pd.read_csv("womens_clothing_reviews.csv")
df = df.dropna(subset=["Review Text"]).reset_index(drop=True)

# ---------- Step 2: Derive sentiment label from Rating ----------
def rating_to_sentiment(r):
    if r <= 2:
        return "NEGATIVE"
    elif r == 3:
        return "NEUTRAL"
    else:
        return "POSITIVE"

df["sentiment"] = df["Rating"].apply(rating_to_sentiment)
print("Class distribution:\n", df["sentiment"].value_counts(), "\n")

# ---------- Step 3: Train/Test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["Review Text"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# ---------- Step 4: TF-IDF + Logistic Regression ----------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# ---------- Step 5: Real evaluation ----------
y_pred = model.predict(X_test_vec)
labels = list(model.classes_)

cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
report = classification_report(y_test, y_pred, labels=labels, output_dict=True)
acc = accuracy_score(y_test, y_pred)

metrics = {
    "dataset": "Women's Clothing E-Commerce Reviews (23,486 real customer reviews)",
    "label_source": "Derived from Rating: 1-2=NEGATIVE, 3=NEUTRAL, 4-5=POSITIVE",
    "labels": labels,
    "confusion_matrix": cm,
    "classification_report": report,
    "accuracy": acc,
    "note": "Computed from a real 80/20 train/test split on actual customer reviews. "
            "class_weight='balanced' used because real ratings are skewed toward positive. "
            "DistilBERT row in the comparison table is NOT measured on this dataset — "
            "it is the model's publicly reported SST-2 benchmark accuracy, shown for "
            "architecture comparison only.",
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, "clothing_sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print(f"Accuracy: {acc:.3f}")
print("Confusion matrix:", cm)
print("\nPer-class results:")
for lbl in labels:
    r = report[lbl]
    print(f"  {lbl}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1-score']:.3f} support={int(r['support'])}")
print("\nSaved: clothing_sentiment_model.pkl, tfidf_vectorizer.pkl, metrics.json")
