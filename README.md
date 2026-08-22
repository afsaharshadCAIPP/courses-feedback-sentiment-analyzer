# ⚡ Customer Feedback Sentiment Analyzer

A high-performance, multi-engine Sentiment Intelligence & Explainable AI (XAI) web application built using **Streamlit**, **Scikit-Learn**, and **Hugging Face Transformers (DistilBERT)**.

This platform allows users to analyze customer product feedback in real-time, inspect word-level prediction drivers via Explainable AI, upload bulk feedback CSVs for batch analytics, and evaluate deep learning vs. traditional ML architecture benchmarks.

---

## 🌟 Key Features

* **🎯 Live Workspace**: Real-time sentiment prediction (Positive, Negative, Neutral) with confidence probability distribution.
* **🔍 Explainable AI (XAI)**: Feature-weight breakdown displaying which specific words drove the classification decision (fixed to use the correct predicted class, not always "Positive").
* **📊 Batch CSV Analytics**: Bulk processing for customer review datasets with automated summary metrics and distribution charts.
* **⚔️ Architecture Benchmarks**: Comparative analysis between **TF-IDF + Logistic Regression** and **DistilBERT Transformer**, with REAL confusion matrix and F1-Score metrics (not hardcoded placeholders).
* **🧩 Aspect-Based Insights**: Goes beyond one label per review — breaks feedback into Fabric/Quality, Fit/Sizing, Price/Value, Delivery/Shipping, and Color/Appearance, and scores each separately. Surfaces which specific aspect drives the most negative feedback, so the "why" behind a sentiment is visible, not just the "what."
* **🎨 High-Contrast UI**: Modern dark-themed dashboard with glassmorphism card layouts and custom product category badges.

---

## 🛠️ Tech Stack

* **Frontend / Dashboard**: Streamlit, HTML5, CSS3
* **Machine Learning**: Scikit-Learn, Joblib
* **Deep Learning (NLP)**: Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`), PyTorch
* **Data Visualization**: Plotly Express, Plotly Graph Objects, Pandas, NumPy

---

## 📊 Dataset & Model Training

The TF-IDF + Logistic Regression model is trained on the **real** "Women's Clothing
E-Commerce Reviews" dataset (23,486 customer reviews). Sentiment labels are derived
from the star rating: 1-2★ = NEGATIVE, 3★ = NEUTRAL, 4-5★ = POSITIVE.

Real evaluation metrics (confusion matrix, precision/recall/F1) are computed on a
held-out 20% test split and saved to `metrics.json` — the app reads these directly,
so the "Architecture Benchmarks" tab always reflects genuine, reproducible results
rather than placeholder numbers.

To retrain from scratch:
```bash
python train_model_real.py
```

## 📂 Project Structure

```text
├── app.py                            # Main Streamlit application file
├── aspect_analyzer.py                # Aspect-based sentiment logic (fabric/fit/price/delivery/color)
├── train_model_real.py               # Trains model on real dataset, generates metrics.json
├── womens_clothing_reviews.csv       # Real dataset (23,486 reviews)
├── metrics.json                      # Real evaluation metrics (used by Tab 4)
├── clothing_sentiment_model.pkl      # Trained Logistic Regression model
├── tfidf_vectorizer.pkl             # Trained TF-IDF Vectorizer
├── requirements.txt                  # Python dependencies list
└── README.md                         # Project documentation
