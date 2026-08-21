# ⚡ Customer Feedback Sentiment Analyzer

A high-performance, multi-engine Sentiment Intelligence & Explainable AI (XAI) web application built using **Streamlit**, **Scikit-Learn**, and **Hugging Face Transformers (DistilBERT)**.

This platform allows users to analyze customer product feedback in real-time, inspect word-level prediction drivers via Explainable AI, upload bulk feedback CSVs for batch analytics, and evaluate deep learning vs. traditional ML architecture benchmarks.

---

## 🌟 Key Features

* **🎯 Live Workspace**: Real-time sentiment prediction (Positive, Negative, Neutral) with confidence probability distribution.
* **🔍 Explainable AI (XAI)**: Feature-weight breakdown displaying which specific words driven the classification decision.
* **📊 Batch CSV Analytics**: Bulk processing for customer review datasets with automated summary metrics and distribution charts.
* **⚔️ Architecture Benchmarks**: Comparative analysis between **TF-IDF + Logistic Regression** and **DistilBERT Transformer**, including Confusion Matrix and F1-Score classification metrics.
* **🎨 High-Contrast UI**: Modern dark-themed dashboard with glassmorphism card layouts and custom product category badges.

---

## 🛠️ Tech Stack

* **Frontend / Dashboard**: Streamlit, HTML5, CSS3
* **Machine Learning**: Scikit-Learn, Joblib
* **Deep Learning (NLP)**: Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`), PyTorch
* **Data Visualization**: Plotly Express, Plotly Graph Objects, Pandas, NumPy

---

## 📂 Project Structure

```text
├── app.py                            # Main Streamlit application file
├── clothing_sentiment_model.pkl      # Trained Logistic Regression model
├── tfidf_vectorizer.pkl             # Trained TF-IDF Vectorizer
├── requirements.txt                  # Python dependencies list
└── README.md                         # Project documentation
