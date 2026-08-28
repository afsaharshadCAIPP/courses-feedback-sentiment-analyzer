# Course Feedback Sentiment Analyzer

A sentiment analysis tool for course reviews. Classifies reviews as
**positive**, **neutral**, or **negative**, and extracts which aspect
of the course (content, instructor, pacing, assignments, structure)
each review is talking about.

Two models are included:
1. **TF-IDF + Logistic Regression** — fast baseline, works well on English, weaker on non-English reviews.
2. **Fine-tuned multilingual DistilBERT** (`distilbert-base-multilingual-cased`) — handles 100+ languages natively, since the dataset contains reviews in English, French, Chinese, Spanish, and more. See `multilingual_distilbert_sentiment.ipynb` (requires a GPU — run on Google Colab with a T4 runtime).

## Dataset
`course_reviews.csv` — 140,320 course reviews with 1-5 star ratings,
spanning multiple languages (English, French, Chinese, Spanish, and more).

Sentiment labels are derived from the star rating:
- 1-2 stars -> negative
- 3 stars -> neutral
- 4-5 stars -> positive

## Files
| File | Purpose |
|---|---|
| `train_model_real.py` | Cleans the data, trains a TF-IDF + Logistic Regression classifier, saves the model and metrics |
| `aspect_analyzer.py` | Keyword-based extraction of which course aspect a review discusses |
| `app.py` | Flask API that serves predictions from the trained model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `course_sentiment_model.pkl` | Trained Logistic Regression sentiment classifier |
| `metrics.json` | Evaluation results (precision/recall/F1, confusion matrix) |
| `requirements.txt` | Python dependencies |
| `course_reviews.csv` | Training data |
| `multilingual_distilbert_sentiment.ipynb` | Colab notebook: fine-tunes multilingual DistilBERT on the same data |

## Results
Overall accuracy: **89.2%**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Negative | 0.455 | 0.622 | 0.526 |
| Neutral | 0.243 | 0.469 | 0.320 |
| Positive | 0.980 | 0.923 | 0.951 |

Full details in `metrics.json`.

## How to run

```bash
pip install -r requirements.txt

# train the model (regenerates the .pkl files and metrics.json)
python train_model_real.py

# start the API
python app.py
```

Then send a request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "The instructor was great but the pacing was too fast."}'
```

## Running the multilingual DistilBERT model
1. Open `multilingual_distilbert_sentiment.ipynb` in Google Colab.
2. `Runtime -> Change runtime type -> T4 GPU`.
3. Run all cells. Upload `course_reviews.csv` when prompted.
4. Training takes roughly 30-60 minutes on a T4 GPU for 3 epochs over the full dataset.

If you hit `ImportError: cannot import name 'VideoReader' from 'torchvision.io'`,
run `!pip uninstall -y torchvision -q` before installing the other packages —
torchvision isn't needed for this text-only task.

## Notes
- The TF-IDF model is a fast baseline. It works well on English text but is
  less reliable on non-English reviews, since TF-IDF treats words in different
  languages as unrelated — this is why the multilingual DistilBERT model is
  included as the stronger alternative for this dataset.
- The `neutral` class has the lowest F1-score in the TF-IDF model, largely
  because it has far fewer training examples than the `positive` class.
