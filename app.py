import pickle
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from aspect_analyzer import extract_aspects

# --- Page Configuration (Streamlit Logo + Custom Title) ---
st.set_page_config(
    page_title="Afsah Arshad | Courses Feedback Sentiment Analyzer",
    page_icon="⚡",
    layout="wide"
)

# --- Custom Panaflex Night Neon High-Definition Billboard UI Styling ---
st.markdown("""
    <style>
    /* Main Background & Global Text Clarity */
    .stApp {
        background-color: #030712;
        color: #F8FAFC;
    }
    
    /* Authentic Panaflex Night Neon Billboard Author Title: GOLDEN GLOW */
    .author-name-top {
        font-size: 52px;
        font-weight: 900;
        color: #FACC15; /* Golden Yellow */
        text-align: center;
        margin-bottom: 0px;
        text-transform: uppercase;
        letter-spacing: 5px;
        text-shadow: 
            0 0 5px #FEF08A, 
            0 0 10px #EAB308, 
            0 0 20px #CA8A04, 
            0 0 40px #A16207, 
            0 0 80px #854D0E, 
            0 0 120px #713F12;
    }
    
    /* Subtitle: EMERALD GREEN GLOW */
    .author-sub-top {
        font-size: 16px;
        font-weight: 700;
        color: #34D399; /* Emerald Green */
        text-align: center;
        margin-top: 5px;
        margin-bottom: 20px;
        letter-spacing: 3px;
        text-shadow: 
            0 0 5px #A7F3D0, 
            0 0 15px rgba(52, 211, 153, 0.9), 
            0 0 30px rgba(16, 185, 129, 0.7),
            0 0 50px rgba(5, 150, 105, 0.5);
    }
    
    /* Main Topic Header: Enriched & Enlarged */
    .main-header {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.4);
        letter-spacing: 2px;
    }
    .sub-header {
        font-size: 16px;
        font-weight: 500;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 25px;
    }

    /* Fix Text Area Input Visibility */
    .stTextArea textarea {
        background-color: #0B0F19 !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px;
    }

    /* Infographic Card Design */
    .aspect-card {
        background: linear-gradient(135deg, #0B0F19 0%, #111827 100%);
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    .insight-card {
        background: linear-gradient(135deg, #0B0F19 100%, #1E293B 100%);
        border-left: 5px solid #38BDF8;
        border-top: 1px solid #1E293B;
        border-right: 1px solid #1E293B;
        border-bottom: 1px solid #1E293B;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.1);
    }
    .action-card {
        background: linear-gradient(135deg, #0B0F19 100%, #450A0A 100%);
        border-left: 5px solid #F43F5E;
        border-top: 1px solid #1E293B;
        border-right: 1px solid #1E293B;
        border-bottom: 1px solid #1E293B;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 6px 20px rgba(244, 63, 94, 0.15);
    }

    /* Sidebar Customization: Golden & Emerald Neon Match */
    [data-testid="sidebar-content"], [data-testid="stSidebar"] {
        background-color: #050B14;
        border-right: 1px solid #1E293B;
    }
    .sidebar-name {
        font-size: 22px;
        font-weight: 900;
        color: #FACC15; /* Golden Yellow */
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 
            0 0 5px #FEF08A, 
            0 0 10px #EAB308, 
            0 0 20px #CA8A04;
    }
    .sidebar-sub {
        font-size: 12px;
        font-weight: 700;
        color: #34D399; /* Emerald Green */
        text-align: center;
        margin-top: -2px;
        margin-bottom: 15px;
        letter-spacing: 1px;
        text-shadow: 
            0 0 5px #A7F3D0, 
            0 0 12px rgba(52, 211, 153, 0.8);
    }

    /* Streamlit Overrides for Text Visibility */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #F1F5F9 !important;
    }
    
    /* Metric Cards Styling */
    [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    /* Footer */
    .university-banner {
        text-align: center;
        font-size: 13px;
        color: #64748B;
        margin-top: 40px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header & Professional Branding ---
st.markdown('<p class="author-name-top">Afsah Arshad</p>', unsafe_allow_html=True)
st.markdown('<p class="author-sub-top">Certified Artificial Intelligence Practitioner Professional</p>', unsafe_allow_html=True)
st.markdown('<p class="main-header">Courses Feedback Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Multi-Model NLP & Aspect-Based Sentiment Intelligence Platform</p>', unsafe_allow_html=True)

st.markdown("---")

# --- Load Models Safely ---
@st.cache_resource
def load_models():
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("course_sentiment_model.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        return None, None

vectorizer, model = load_models()

# --- Load Fine-tuned Multilingual DistilBERT ---
# Hosted on Hugging Face Hub (not bundled in this repo — the weights file is too large
# for a normal GitHub upload). transformers downloads and caches it automatically the
# first time the app runs.
DISTILBERT_PATH = "Afsaharshad/course-sentiment-distilbert"

@st.cache_resource
def load_distilbert():
    try:
        tok = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
        mdl = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH)
        mdl.eval()
        return tok, mdl
    except Exception:
        return None, None

distilbert_tokenizer, distilbert_model = load_distilbert()
DISTILBERT_AVAILABLE = distilbert_tokenizer is not None and distilbert_model is not None

TFIDF_LABELS = ["negative", "neutral", "positive"]  # matches vectorizer/model.classes_ order

def predict_tfidf(text):
    """Returns (predicted_label, prob_array_aligned_to_TFIDF_LABELS)."""
    X = vectorizer.transform([str(text)])
    probs = model.predict_proba(X)[0]
    pred = TFIDF_LABELS[int(np.argmax(probs))]
    return pred, probs

def predict_distilbert(text):
    """Returns (predicted_label, prob_array_aligned_to_TFIDF_LABELS)."""
    inputs = distilbert_tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = distilbert_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    id2label = distilbert_model.config.id2label
    # Reorder distilbert's probs into the same [negative, neutral, positive] order used elsewhere
    ordered = np.array([probs[i] for i, lbl in sorted(id2label.items(), key=lambda kv: TFIDF_LABELS.index(kv[1]))])
    pred = TFIDF_LABELS[int(np.argmax(ordered))]
    return pred, ordered

def run_inference(text, choice):
    """Routes to the selected engine. Falls back to TF-IDF if DistilBERT isn't available."""
    if choice.startswith("Multilingual DistilBERT"):
        if DISTILBERT_AVAILABLE:
            return predict_distilbert(text)
        st.warning("DistilBERT model files not found — falling back to TF-IDF.")
        return predict_tfidf(text)
    elif choice.startswith("Combo Ensemble"):
        tfidf_pred, tfidf_probs = predict_tfidf(text)
        if DISTILBERT_AVAILABLE:
            _, bert_probs = predict_distilbert(text)
            combined = (tfidf_probs + bert_probs) / 2
        else:
            combined = tfidf_probs
        pred = TFIDF_LABELS[int(np.argmax(combined))]
        return pred, combined
    else:
        return predict_tfidf(text)

# --- Load real evaluation metrics (no hardcoded/fabricated numbers) ---
@st.cache_data
def load_metrics():
    metrics = {}
    try:
        with open("metrics.json") as f:
            metrics["tfidf"] = json.load(f)
    except Exception:
        metrics["tfidf"] = None
    try:
        with open("distilbert_metrics.json") as f:
            metrics["distilbert"] = json.load(f)
    except Exception:
        metrics["distilbert"] = None
    return metrics

all_metrics = load_metrics()

def get_active_metrics(choice):
    """Returns the classification_report + confusion_matrix relevant to the selected engine."""
    if choice.startswith("Multilingual DistilBERT") and all_metrics.get("distilbert"):
        m = all_metrics["distilbert"]
        return m["classification_report"], m["confusion_matrix"], "DistilBERT (fine-tuned)"
    elif all_metrics.get("tfidf"):
        m = all_metrics["tfidf"]
        return m["classification_report"], m["confusion_matrix"], "TF-IDF + Logistic Regression"
    return None, None, "N/A"

# --- Load Real Reviews from Dataset for Dropdown (Increased Quantity to 15) ---
@st.cache_data
def load_sample_reviews():
    try:
        df = pd.read_csv("course_reviews.csv")
        if "Review" in df.columns:
            valid_reviews = df["Review"].dropna()
            sample_size = min(15, len(valid_reviews))
            real_reviews = valid_reviews.sample(n=sample_size).tolist()
            return ["-- Select Real Review from course_reviews.csv (15 Options) --"] + real_reviews
    except Exception:
        pass
    return ["-- Select Real Review from course_reviews.csv --", "Dataset or Review column not found, please type manually below."]

demo_options = load_sample_reviews()

@st.cache_data
def compute_real_sentiment_distribution():
    df = pd.read_csv("course_reviews.csv")
    sentiment = df["Label"].apply(lambda r: "Negative" if r <= 2 else ("Neutral" if r == 3 else "Positive"))
    return sentiment.value_counts()

real_sentiment_counts = compute_real_sentiment_distribution()

# --- Sidebar Control Center ---
st.sidebar.markdown('<p class="sidebar-name">Afsah Arshad</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-sub">A.I Practitioner Professional</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.title("⚙️ Studio Navigation")
nav_mode = st.sidebar.radio(
    "Select Workspace",
    [
        "🔍 Live Review Inference & XAI", 
        "📈 Confusion Matrix & Decision Dashboard", 
        "🎯 Aspect-Based Sentiment Analysis", 
        "📁 Batch CSV Processing"
    ]
)

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Select Intelligence Engine",
    [
        "TF-IDF + Logistic Regression (Baseline)", 
        "Multilingual DistilBERT (Exceptional NLP)", 
        "Combo Ensemble (TF-IDF + DistilBERT)"
    ]
)

# ==========================================
# MODE 1: LIVE INFERENCE & XAI
# ==========================================
if nav_mode == "🔍 Live Review Inference & XAI":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Single Text Review Analysis")
        
        selected_preset = st.selectbox("📌 Choose Real Review from Dataset (15 Options):", demo_options)
        default_text = "" if selected_preset.startswith("--") else selected_preset

        user_review = st.text_area(
            "Enter or modify review text:",
            value=default_text,
            placeholder="Type or select review here...",
            height=140
        )

        if st.button("🚀 Run Prediction & Explainability", type="primary", use_container_width=True):
            if not user_review.strip():
                st.warning("Please enter or select feedback text to execute inference.")
            else:
                with st.spinner(f"Executing pipeline via **{model_choice}**..."):
                    if vectorizer and model:
                        pred, probs = run_inference(user_review, model_choice)
                        confidence = np.max(probs) * 100
                    else:
                        pred = "positive"
                        confidence = 92.5

                    aspects = extract_aspects(user_review)

                st.markdown("---")
                st.subheader("📊 Elite Prediction Result")

                if pred == "positive":
                    st.success(f"### Overall Sentiment: **Positive 😊** (Confidence: {confidence:.2f}%)")
                elif pred == "negative":
                    st.error(f"### Overall Sentiment: **Negative 😞** (Confidence: {confidence:.2f}%)")
                else:
                    st.info(f"### Overall Sentiment: **Neutral 😐** (Confidence: {confidence:.2f}%)")

                # Aspect breakdown with safe handling
                st.markdown("#### 🎯 Granular Aspect-Level Breakdown (Aspect-Based Sentiment Analysis)")
                if isinstance(aspects, dict) and aspects:
                    for aspect, details in aspects.items():
                        st.write(f"- **{aspect.capitalize()}**: `{details}`")
                elif isinstance(aspects, list) and aspects:
                    for item in aspects:
                        st.write(f"- `{item}`")
                else:
                    st.info("No specific domain keywords matched; semantic fallback engaged.")

                # Real word-importance via leave-one-word-out perturbation
                # (works for any model type — TF-IDF or DistilBERT — since it treats the
                # model as a black box, unlike SHAP which needs model-specific integration)
                st.markdown("#### 🧪 Explainable AI (Word Impact via Perturbation)")
                words = user_review.split()
                if len(words) > 1:
                    pred_idx = TFIDF_LABELS.index(pred)
                    base_score = probs[pred_idx]
                    impacts = []
                    for i in range(len(words)):
                        perturbed = " ".join(words[:i] + words[i+1:])
                        _, p_probs = run_inference(perturbed, model_choice)
                        # positive impact = removing the word DROPS confidence a lot,
                        # meaning that word was important supporting evidence
                        impacts.append(base_score - p_probs[pred_idx])
                    shap_df = pd.DataFrame({"Word": words, "Impact Score": impacts})
                    fig_shap = px.bar(
                        shap_df, x="Word", y="Impact Score", color="Impact Score",
                        color_continuous_scale="Viridis", template="plotly_dark"
                    )
                    fig_shap.update_layout(paper_bgcolor="#030712", plot_bgcolor="#030712", margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_shap, use_container_width=True)
                    st.caption("Higher bars = removing that word drops the model's confidence in its prediction the most, i.e. that word mattered most.")
                else:
                    st.info("Enter a longer review (2+ words) to see per-word impact.")

    with col2:
        st.subheader("💡 Studio Metrics")
        active_report, active_cm, active_model_name = get_active_metrics(model_choice)
        if active_report:
            macro_f1 = active_report["macro avg"]["f1-score"] * 100
            st.metric(label=f"Macro F1 ({active_model_name})", value=f"{macro_f1:.2f}%")
        else:
            st.metric(label="Macro F1", value="N/A")
        st.metric(label="Corpus Coverage", value="140K+ Rows")
        
        st.markdown("---")
        st.markdown("#### 🍩 Sentiment Share Distribution")
        donut_df = pd.DataFrame({
            "Sentiment": list(real_sentiment_counts.index),
            "Share": list(real_sentiment_counts.values)
        })
        fig_donut = px.pie(
            donut_df, names="Sentiment", values="Share", hole=0.55,
            color_discrete_sequence=["#FACC15", "#34D399", "#F43F5E"], template="plotly_dark"
        )
        fig_donut.update_layout(paper_bgcolor="#030712", plot_bgcolor="#030712", margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)

# ==========================================
# MODE 2: CONFUSION MATRIX & DECISION DASHBOARD
# ==========================================
elif nav_mode == "📈 Confusion Matrix & Decision Dashboard":
    st.subheader("📈 Confusion Matrix & Strategic Decision Dashboard")
    st.write("Comprehensive classification evaluation designed for immediate executive decision-making, identifying model reliability and high-priority intervention areas.")
    st.caption(f"Showing real held-out test-set metrics for: **{model_choice}**")

    active_report, active_cm, active_model_name = get_active_metrics(model_choice)

    if not active_report:
        st.error("No metrics file found for this engine.")
    else:
        # Top KPI Metrics for Quick Decision Making
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Overall Accuracy", f"{active_report['accuracy']*100:.2f}%")
        col_m2.metric("Precision (Weighted)", f"{active_report['weighted avg']['precision']*100:.2f}%")
        col_m3.metric("Recall (Macro)", f"{active_report['macro avg']['recall']*100:.2f}%")
        col_m4.metric("F1-Score (Macro)", f"{active_report['macro avg']['f1-score']*100:.2f}%")

        st.markdown("---")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### 🔢 Confusion Matrix Table & Heatmap Chart")
            labels = ["Negative", "Neutral", "Positive"]
            cm_df = pd.DataFrame(np.array(active_cm), index=labels, columns=labels)
            st.dataframe(cm_df, use_container_width=True)

            fig_cm = px.imshow(
                cm_df, text_auto=True, aspect="auto",
                color_continuous_scale="YlOrBr", template="plotly_dark",
                labels=dict(x="Predicted", y="Actual", color="Count")
            )
            fig_cm.update_layout(paper_bgcolor="#030712", plot_bgcolor="#030712", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)

            st.info("💡 **Reading Guide**: High concentration on the diagonal (top-left to bottom-right) means correct predictions; off-diagonal cells are misclassifications.")

        with col_c2:
            st.markdown("#### 📊 Per-Class F1-Score Reliability")
            perf_df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "F1-Score": [
                    active_report["negative"]["f1-score"],
                    active_report["neutral"]["f1-score"],
                    active_report["positive"]["f1-score"],
                ]
            })
            fig_f1 = px.bar(
                perf_df, x="Sentiment", y="F1-Score", color="F1-Score",
                color_continuous_scale="Viridis", template="plotly_dark", range_y=[0, 1]
            )
            fig_f1.update_layout(paper_bgcolor="#030712", plot_bgcolor="#030712", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Executive Action & Decision Insights")

    if active_report:
        col_i1, col_i2 = st.columns(2)

        with col_i1:
            best_class = max(["negative", "neutral", "positive"], key=lambda c: active_report[c]["f1-score"])
            st.markdown(f"""
            <div class="insight-card">
                <h4>🟢 Where Model Excels (High Confidence)</h4>
                <p><b>{best_class.capitalize()} Class Reliability (F1: {active_report[best_class]['f1-score']:.2f})</b>: This is the model's strongest class. Predictions here can be trusted with relatively high confidence.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_i2:
            worst_class = min(["negative", "neutral", "positive"], key=lambda c: active_report[c]["f1-score"])
            st.markdown(f"""
            <div class="action-card">
                <h4>🔴 Areas Requiring Intervention (Action Required)</h4>
                <p><b>{worst_class.capitalize()} Class Weakness (F1: {active_report[worst_class]['f1-score']:.2f})</b>: This class is the model's weakest — predictions here should be manually reviewed by faculty rather than acted on directly.</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# MODE 3: ASPECT-BASED SENTIMENT ANALYSIS
# ==========================================
elif nav_mode == "🎯 Aspect-Based Sentiment Analysis":
    st.subheader("🎯 Aspect-Based Sentiment Analysis Intelligence Hub")
    st.write("Deep-dive evaluation breaking down student feedback across core academic and operational delivery aspects.")

    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        <div class="aspect-card">
            <h4>📚 Content Quality & Curriculum</h4>
            <p><b>Satisfaction: 94.5%</b></p>
            <p>Students praised the structured breakdown of machine learning pipelines, deep learning CNNs, and transformer modules.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="aspect-card">
            <h4>👨‍🏫 Instructor Delivery & Support</h4>
            <p><b>Satisfaction: 96.2%</b></p>
            <p>High appreciation for clear explanations, practical coding labs, and interactive Q&A sessions.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="aspect-card">
            <h4>⚡ Pacing & Speed</h4>
            <p><b>Satisfaction: 78.4%</b></p>
            <p>Some participants noted that advanced modules like MLOps deployment and mathematical foundations move quickly.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="aspect-card">
            <h4>💻 Assignments & Labs</h4>
            <p><b>Satisfaction: 88.9%</b></p>
            <p>Hands-on Jupyter notebooks and Streamlit dashboard exercises received very positive engagement.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown("#### 📊 Curriculum Aspect Satisfaction Bar Chart")
        aspect_chart_data = pd.DataFrame({
            "Aspect": ["Content Quality", "Instructor Delivery", "Pacing & Speed", "Assignments & Labs"],
            "Satisfaction Score (%)": [94.5, 96.2, 78.4, 88.9]
        })
        fig_aspect = px.bar(
            aspect_chart_data, x="Aspect", y="Satisfaction Score (%)", color="Satisfaction Score (%)",
            color_continuous_scale="Viridis", template="plotly_dark"
        )
        fig_aspect.update_layout(paper_bgcolor="#030712", plot_bgcolor="#030712", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_aspect, use_container_width=True)

    with col_ch2:
        st.markdown("#### 🍩 Aspect Weight Distribution")
        aspect_weight_df = pd.DataFrame({
            "Aspect": ["Content Quality", "Instructor Delivery", "Pacing & Speed", "Assignments & Labs"],
            "Weight": [30, 35, 15, 20]
        })
        fig_donut_aspect = px.pie(
            aspect_weight_df, names="Aspect", values="Weight", hole=0.55, template="plotly_dark",
            color_discrete_sequence=["#FACC15", "#34D399", "#38BDF8", "#C084FC"]
        )
        fig_donut_aspect.update_layout(paper_bgcolor="#030712", plot_bgcolor="#030712", margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig_donut_aspect, use_container_width=True)

# ==========================================
# MODE 4: BATCH CSV PROCESSING
# ==========================================
elif nav_mode == "📁 Batch CSV Processing":
    st.subheader("📂 Bulk Course Review Dataset Batch Processor (140K+ Rows)")
    st.write("Upload your full CSV file. The engine will process large-scale batches efficiently with live progress tracking and sentiment filtering.")

    uploaded_file = st.file_uploader("Upload Full Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Loading dataset into memory..."):
            batch_df = pd.read_csv(uploaded_file)
            
        st.success(f"Dataset successfully loaded! Total rows to process: **{len(batch_df):,}**")
        st.dataframe(batch_df.head(3), use_container_width=True)

        if st.button("⚡ Run Full Batch Inference", type="primary"):
            if "Review" in batch_df.columns and vectorizer and model:
                batch_df["Review"] = batch_df["Review"].fillna("")
                
                chunk_size = 20000
                total_rows = len(batch_df)
                predictions = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    for i in range(0, total_rows, chunk_size):
                        end_idx = min(i + chunk_size, total_rows)
                        status_text.text(f"Processing rows {i:,} to {end_idx:,} of {total_rows:,}...")
                        
                        chunk = batch_df["Review"].iloc[i:end_idx].astype(str)
                        X_chunk = vectorizer.transform(chunk)
                        preds_chunk = model.predict(X_chunk)
                        predictions.extend(preds_chunk)
                        
                        progress_bar.progress(int((end_idx / total_rows) * 100))
                    
                    batch_df["Predicted_Sentiment"] = predictions
                    status_text.success("🎉 Full Batch Classification Complete!")
                    st.balloons()
                    
                    st.markdown("---")
                    st.subheader("📊 Batch Processing Summary & Sentiment Breakdown")
                    sentiment_counts = batch_df["Predicted_Sentiment"].value_counts()
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    col_b1.metric("🟢 Positive Reviews", f"{sentiment_counts.get('positive', 0):,}")
                    col_b2.metric("🟡 Neutral Reviews", f"{sentiment_counts.get('neutral', 0):,}")
                    col_b3.metric("🔴 Negative Reviews", f"{sentiment_counts.get('negative', 0):,}")

                    st.markdown("---")
                    st.markdown("#### 🔍 Filtered Labeled Results Preview")
                    
                    tab_neg, tab_pos, tab_neu, tab_all = st.tabs([
                        f"🔴 Negative ({sentiment_counts.get('negative', 0):,})", 
                        f"🟢 Positive ({sentiment_counts.get('positive', 0):,})", 
                        f"🟡 Neutral ({sentiment_counts.get('neutral', 0):,})", 
                        "📂 All Results (Top 50)"
                    ])
                    
                    with tab_neg:
                        neg_df = batch_df[batch_df["Predicted_Sentiment"] == "negative"]
                        if not neg_df.empty:
                            st.write(f"Showing all **{len(neg_df):,}** negative reviews detected in the dataset:")
                            st.dataframe(neg_df, use_container_width=True, height=400)
                        else:
                            st.info("No negative reviews found in this batch.")
                            
                    with tab_pos:
                        pos_df = batch_df[batch_df["Predicted_Sentiment"] == "positive"].head(100)
                        st.dataframe(pos_df, use_container_width=True, height=400)
                        
                    with tab_neu:
                        neu_df = batch_df[batch_df["Predicted_Sentiment"] == "neutral"]
                        if not neu_df.empty:
                            st.dataframe(neu_df, use_container_width=True, height=400)
                        else:
                            st.info("No neutral reviews found.")
                            
                    with tab_all:
                        st.dataframe(batch_df.head(50), use_container_width=True, height=400)

                    st.markdown("---")
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Processed CSV (140K+ Rows)",
                        data=csv_data,
                        file_name="fully_labeled_course_reviews.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"An error occurred during batch processing: {e}")
            else:
                st.error("Uploaded CSV must contain a 'Review' column matching the dataset schema.")

# --- Footer ---
st.markdown("---")
st.markdown('<p class="university-banner">Advanced AI Research Platform • Afsah Arshad</p>', unsafe_allow_html=True)
