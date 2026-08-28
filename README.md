<div align="center">

# ⚡ Customer Feedback Sentiment Analyzer & Aspect Intelligence Platform
### *Advanced Multi-Model NLP, Deep Learning & Explainable AI Solution*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%2520Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DistilBERT-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%2520Visualizations-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

</div>

---

## 👩‍💻 Author & Developer
* **Name:** Afsah Arshad
* **Certification:** Certified Artificial Intelligence Practitioner Professional

---

## 🚀 Overview
The **Customer Feedback Sentiment Analyzer** is an enterprise-grade Natural Language Processing (NLP) web application designed to analyze academic and course feedback with exceptional accuracy. Powered by multi-model machine learning classifiers, transformer-based deep learning architectures, and **Explainable AI (XAI)**, this platform bridges raw textual feedback with actionable institutional insights.

---

## 🤖 Intelligence Engines & Model Architecture

Users can dynamically switch between three sophisticated NLP models depending on latency requirements and text complexity:

### 1. 📊 TF-IDF + Logistic Regression (Baseline Model)
* **Architecture:** Term Frequency-Inverse Document Frequency (TF-IDF) n-gram vectorizer paired with a tuned Linear Logistic Regression classifier.
* **How it Functions:**
  1. Converts raw text into numerical n-gram token frequencies, penalizing universally common words and emphasizing rare domain-specific terms.
  2. The Logistic Regression model applies learned weight vectors to calculate class log-odds, outputting calibrated probability scores across Positive, Neutral, and Negative classes.
* **Best Used For:** High-speed, low-latency bulk dataset processing and clean English reviews.

### 2. 🌐 Multilingual DistilBERT (Advanced Transformer Model)
* **Why Multilingual over Normal (Monolingual) Models?**
  * Standard or monolingual English models (`distilbert-base-uncased`) fail when students write reviews using mixed languages, Roman Urdu, slang, or localized syntax (e.g., *"Course ka content bohot acha tha but pacing thori fast thi"*). 
  * A **Multilingual Transformer (`distilbert-base-multilingual-cased`)** is pre-trained across 104 languages, allowing it to capture semantic representations, contextual nuances, tone, and multi-language phrasing seamlessly without losing meaning.
* **How it Functions:**
  1. Utilizes self-attention mechanisms to weigh the importance of surrounding words in a sentence dynamically.
  2. Maps input tokens into a multi-dimensional semantic embedding space where sentiment polarity is recognized by contextual relationships rather than fixed keywords.

### 3. ⚡ Combo Ensemble (TF-IDF + DistilBERT Hybrid Engine)
* **Architecture:** Soft-voting hybrid ensemble combining statistical representations with deep transformer embeddings.
* **How it Functions:**
  1. The input review is processed simultaneously through both the TF-IDF statistical vectorizer pipeline and the DistilBERT attention layers.
  2. The pipeline calculates a weighted average of both models' predicted class probability vectors:
     $$\text{Probability}_{\text{Combo}} = \alpha \cdot P_{\text{DistilBERT}} + (1 - \alpha) \cdot P_{\text{TF-IDF}}$$
  3. This combined approach neutralizes individual model errors, reduces variance, and maximizes generalization performance on unseen or complex feedback.

---

## ✨ Key Features & Capabilities

1. **🔍 Live Review Inference & Explainability (XAI):**
   * Instant sentiment classification (Positive, Neutral, Negative) with confidence probability outputs.
   * Granular Aspect-Based Sentiment Analysis (ABSA) breaking down feedback across core curriculum and delivery pillars.
   * Explainable AI (SHAP) feature attribution visualization highlighting key words driving predictions.
2. **📈 Confusion Matrix & Decision Dashboard:**
   * Comprehensive performance metrics (Precision, Recall, Macro F1-Score: **90.77%**, Accuracy).
   * Executive decision insights highlighting high-confidence areas and boundary misclassifications for strategic intervention.
3. **🎯 Aspect-Based Intelligence Hub:**
   * Dedicated evaluations across Content Quality, Instructor Delivery, Pacing & Speed, and Hands-on Labs.
4. **📁 Bulk CSV Dataset Processor:**
   * Batch inference support allowing institutions to process raw CSV review files and export structured reports.

---

## 🛠️ Technology Stack
* **Frontend / UI:** Streamlit (Custom Panaflex Neon Dark-Theme CSS & Interactive Plotly Charts)
* **Machine Learning & NLP:** Scikit-Learn, PyTorch, Hugging Face Transformers (DistilBERT), TF-IDF
* **Data & Analytics:** Pandas, NumPy
* **Explainability:** SHAP / Feature Attribution Scoring

---

## ⚙️ Installation & Local Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/customer-feedback-sentiment-analyzer.git](https://github.com/your-username/customer-feedback-sentiment-analyzer.git)
   cd customer-feedback-sentiment-analyzer
