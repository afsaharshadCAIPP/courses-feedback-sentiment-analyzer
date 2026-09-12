# Course Feedback Intelligence — Coursera Sentiment Analyzer

Multilingual sentiment analysis over 137,301 real Coursera course reviews, comparing a classical
TF-IDF + Logistic Regression baseline against a fine-tuned multilingual transformer
(distilbert-base-multilingual-cased).

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application (dashboard, live inference, model comparison, aspect analysis) |
| `aspect_analyzer.py` | Aspect-based sentiment module (course content, instructor, difficulty, etc.) |
| `shap_explainer.py` | Real SHAP-based word-level explainability for the DistilBERT model |
| `coursera_tfidf_logistic_model.pkl` | Trained TF-IDF + Logistic Regression model |
| `coursera_tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `metrics.json` | Real evaluation metrics for the TF-IDF baseline |
| `distilbert_metrics.json` | Real evaluation metrics for the fine-tuned DistilBERT model |
| `reviews_by_course.csv` | Full labelled dataset (140,320 raw rows) |
| `requirements.txt` | Python dependencies |

## Where the DistilBERT model lives

The fine-tuned Multilingual DistilBERT model's weights (~541 MB) are **not** stored in this
repository — they are hosted on Hugging Face Hub at:

**[Afsah-2027/coursera-multilingual-distilbert](https://huggingface.co/Afsah-2027/coursera-multilingual-distilbert)**

`app.py` downloads and caches this model automatically the first time the app runs. This keeps
the GitHub repository lightweight and avoids GitHub's 100 MB file-size limit.

## Models

| Model | Macro F1 | Notes |
|---|---|---|
| TF-IDF + Logistic Regression | 52.5% | Fast baseline, trained on raw multilingual text |
| Multilingual DistilBERT (fine-tuned) | 65.9% | Full 140k dataset, 80/10/10 split, class-weighted loss |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed via Streamlit Community Cloud, connected to this GitHub repository, with `app.py` as
the entry point.
