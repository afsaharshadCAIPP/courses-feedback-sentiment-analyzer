# Courses Feedback Sentiment Analyzer — Multilingual DistilBERT & TF-IDF Dashboard

An enterprise-grade multilingual sentiment analysis and aspect-based insights dashboard built for online course reviews. Developed as part of applied natural language processing workflows, this project bridges classical machine learning baselines with state-of-the-art transformer architectures to parse, analyze, and explain learner feedback at scale.

---

## 🌟 Key Features & Workspaces

The application is structured into four core interactive workspaces designed for deep data exploration:

1. **Live Inference Engine**: 
   * Test real-time sentiment predictions on any custom review or load samples directly from the corpus.
   * Switch dynamically between dual engines: Classical **TF-IDF + Logistic Regression** or the fine-tuned **Multilingual DistilBERT** transformer.
   * **Word-Level Explainability (SHAP)**: Automatically computes token-level contributions using SHAP's Partition explainer to visualize which words push predictions toward or away from specific sentiment classes.

2. **Model Comparison Dashboard**:
   * Comprehensive side-by-side benchmarking of models across Accuracy, Macro F1, and Weighted F1 scores.
   * Per-class performance tracking (`NEGATIVE`, `NEUTRAL`, `POSITIVE`).
   * Interactive Confusion Matrices for error analysis and misclassification tracking.

3. **Aspect-Based Sentiment Analysis (ABSA)**:
   * Moves beyond coarse overall ratings to extract granular feedback dimensions (e.g., course content, instructor, pacing, assignments).
   * Highlights specific verbatim text evidence matching each aspect alongside confidence metrics.

4. **Corpus Overview**:
   * Macro-level distribution views of sentiment across thousands of course reviews.
   * Leaderboards and volume tracking for top-performing or most-reviewed courses.

---

## 📊 Model Performance Benchmarks

| Model Architecture | Macro F1 | Key Training Notes |
| :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 52.5% | Fast baseline, trained on raw multilingual text vectors |
| **Multilingual DistilBERT (Fine-tuned)** | **65.9%** | Full dataset split, 80/10/10 configuration, class-weighted loss optimization |

---

## 🧠 Dual Modeling Engines

* **TF-IDF + Logistic Regression Module**: Utilizes a serialized scikit-learn pipeline with custom vectorization to provide an instant, lightweight text classification baseline.
* **Multilingual DistilBERT Module**: Fine-tuned cross-lingual transformer variant to capture nuanced expressions without sacrificing inference speed.
  * **Hosting & Weights Architecture**: To comply strictly with GitHub’s 100 MB file-size limitations, the heavy fine-tuned transformer weights (~541 MB `model.safetensors`) are hosted and version-controlled externally on the Hugging Face Hub:
    * 👉 **Hugging Face Model Hub**: [Afsah-2027/coursera-multilingual-distilbert](https://huggingface.co/Afsah-2027/coursera-multilingual-distilbert)
  * **Automatic Caching**: The Streamlit application automatically connects to Hugging Face via `transformers` on startup, downloading and caching configuration, tokenizer, and safetensors weights seamlessly during its initial run.

---

## 🛠️ Project Structure & Files

* `app.py`: Main Streamlit multi-tab dashboard script adhering to an ink-and-parchment academic instrument aesthetic.
* `aspect_analyzer.py`: Aspect-based extraction and per-dimension sentiment scoring engine.
* `shap_explainer.py`: Integration script for calculating transformer-based SHAP values.
* `coursera_tfidf_logistic_model.pkl`: Serialized baseline classification model.
* `coursera_tfidf_vectorizer.pkl`: Fitted TF-IDF feature extractor.
* `metrics.json` & `distilbert_metrics.json`: Stored evaluation metrics, classification reports, and confusion matrix arrays.
* `reviews_by_course.csv`: Sampled corpus dataset powering corpus overviews and quick-load options.
* `requirements.txt`: Python package dependencies.

---

## 🚀 Installation & Local Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/afsaharshadCAIPP/courses-feedback-sentiment-analyzer.git](https://github.com/afsaharshadCAIPP/courses-feedback-sentiment-analyzer.git)
   cd courses-feedback-sentiment-analyzer
