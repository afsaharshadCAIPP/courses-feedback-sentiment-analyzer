"""
app.py
Small Flask app that serves the trained course-review sentiment model.

Run:
    python app.py

Then POST to http://localhost:5000/predict with JSON: {"review": "some text"}
"""

import pickle

from flask import Flask, jsonify, request

from aspect_analyzer import extract_aspects

app = Flask(__name__)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("course_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    review = data.get("review", "")

    if not review.strip():
        return jsonify({"error": "review text is required"}), 400

    X = vectorizer.transform([review])
    sentiment = model.predict(X)[0]
    aspects = extract_aspects(review)

    return jsonify(
        {
            "review": review,
            "sentiment": sentiment,
            "aspects": aspects,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
