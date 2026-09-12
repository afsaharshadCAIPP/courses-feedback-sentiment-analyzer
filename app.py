import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Course Feedback Intelligence — Coursera",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# DESIGN SYSTEM — vibrant, premium "AI product" aesthetic
# Deep space-navy background, glassmorphism cards, gradient accents.
# =====================================================================
BG_DEEP = "#0A0E1C"
BG_MID = "#121834"
GLASS = "rgba(255,255,255,0.045)"
GLASS_BORDER = "rgba(255,255,255,0.09)"
TEXT = "#EAEBFA"
MUTED = "#8B93B8"

VIOLET = "#7C5CFF"
CYAN = "#22D3EE"
PINK = "#F472B6"

RUST = "#FB7185"      # negative (vivid rose)
OCHRE = "#FBBF24"     # neutral (vivid amber)
TEAL = "#34D399"      # positive (vivid emerald)

SENTIMENT_COLOR = {"NEGATIVE": RUST, "NEUTRAL": OCHRE, "POSITIVE": TEAL,
                    "negative": RUST, "neutral": OCHRE, "positive": TEAL}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background: radial-gradient(circle at 15% 0%, #1B1F4B 0%, {BG_MID} 35%, {BG_DEEP} 75%);
    color: {TEXT};
}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #10142B 0%, {BG_DEEP} 100%);
    border-right: 1px solid {GLASS_BORDER};
}}
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: {TEXT};
    letter-spacing: -0.01em;
}}
.gradient-text {{
    background: linear-gradient(90deg, {VIOLET}, {CYAN} 60%, {PINK});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.masthead {{
    padding-bottom: 22px;
    margin-bottom: 30px;
    border-bottom: 1px solid {GLASS_BORDER};
}}
.masthead .kicker {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    color: {VIOLET};
    letter-spacing: 0.08em;
    background: rgba(124,92,255,0.12);
    border: 1px solid rgba(124,92,255,0.35);
    border-radius: 999px;
    padding: 5px 14px;
    margin-bottom: 14px;
}}
.masthead h1 {{
    font-size: 42px;
    margin: 4px 0 8px 0;
}}
.masthead .dek {{
    color: {MUTED};
    font-size: 15.5px;
    max-width: 680px;
    line-height: 1.6;
}}
.panel {{
    background: {GLASS};
    backdrop-filter: blur(14px);
    border: 1px solid {GLASS_BORDER};
    border-radius: 18px;
    padding: 24px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}}
.panel-glow-violet {{ box-shadow: 0 8px 30px rgba(124,92,255,0.18), 0 0 0 1px rgba(124,92,255,0.15) inset; }}
.panel-hero {{
    background: linear-gradient(135deg, rgba(124,92,255,0.16), rgba(34,211,238,0.10));
    backdrop-filter: blur(14px);
    border: 1px solid rgba(124,92,255,0.35);
    border-radius: 20px;
    padding: 28px 30px;
    margin-bottom: 20px;
    box-shadow: 0 12px 40px rgba(124,92,255,0.15);
}}
.panel-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    font-weight: 600;
    color: {CYAN};
    letter-spacing: 0.06em;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid {GLASS_BORDER};
}}
.metric-row {{
    display: flex;
    gap: 14px;
    margin-bottom: 22px;
    flex-wrap: wrap;
}}
.metric-cell {{
    flex: 1;
    min-width: 150px;
    background: {GLASS};
    backdrop-filter: blur(14px);
    border: 1px solid {GLASS_BORDER};
    border-radius: 16px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}}
.metric-cell::before {{
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, {VIOLET}, {CYAN});
}}
.metric-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: {MUTED};
    margin-bottom: 8px;
    letter-spacing: 0.04em;
}}
.metric-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 28px;
    color: {TEXT};
}}
.sentiment-badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.04em;
    padding: 7px 16px;
    border-radius: 999px;
    background: currentColor;
}}
.sentiment-badge span {{ color: {BG_DEEP}; }}
.result-verdict {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 36px;
    margin: 12px 0 6px 0;
}}
.evidence-line {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: {MUTED};
    background: rgba(255,255,255,0.03);
    border-left: 3px solid {VIOLET};
    border-radius: 0 8px 8px 0;
    padding: 8px 12px;
    margin: 6px 0;
}}
.pending-note {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: {OCHRE};
    background: rgba(251,191,36,0.08);
    border: 1px dashed {OCHRE};
    border-radius: 12px;
    padding: 14px 16px;
}}
div.stButton > button {{
    background: linear-gradient(90deg, {VIOLET}, #9D7BFF);
    color: white;
    border: none;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    padding: 11px 26px;
    box-shadow: 0 6px 20px rgba(124,92,255,0.35);
    transition: all 0.15s ease;
}}
div.stButton > button:hover {{
    box-shadow: 0 8px 28px rgba(124,92,255,0.55);
    transform: translateY(-1px);
}}
[data-testid="stTextArea"] textarea {{
    background: rgba(255,255,255,0.04);
    color: {TEXT};
    border: 1px solid {GLASS_BORDER};
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
}}
[data-testid="stSelectbox"] > div > div {{
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    border: 1px solid {GLASS_BORDER};
}}
hr {{ border-color: {GLASS_BORDER}; }}
div[role="radiogroup"] label span:first-child {{ border-color: {MUTED} !important; }}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATA / MODEL LOADING
# =====================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
HF_DISTILBERT_REPO = "Afsah-2027/coursera-multilingual-distilbert"
DISTILBERT_AVAILABLE = True

@st.cache_resource
def load_tfidf():
    model = joblib.load(os.path.join(BASE, "coursera_tfidf_logistic_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE, "coursera_tfidf_vectorizer.pkl"))
    return model, vectorizer

@st.cache_data
def load_metrics():
    with open(os.path.join(BASE, "metrics.json")) as f:
        return json.load(f)

@st.cache_data
def load_distilbert_metrics():
    path = os.path.join(BASE, "distilbert_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_dataset_sample(n=25):
    path = os.path.join(BASE, "reviews_by_course.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.sample(n=min(n, len(df)), random_state=7)

tfidf_model, tfidf_vectorizer = load_tfidf()
metrics = load_metrics()
distilbert_metrics = load_distilbert_metrics()
sample_df = load_dataset_sample()

DISTILBERT_LOAD_ERROR = None
if DISTILBERT_AVAILABLE:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import concurrent.futures

    @st.cache_resource
    def load_distilbert():
        def _load():
            tok = AutoTokenizer.from_pretrained(HF_DISTILBERT_REPO)
            mdl = AutoModelForSequenceClassification.from_pretrained(HF_DISTILBERT_REPO)
            mdl.eval()
            return tok, mdl
        # Hard timeout so a network hiccup can't hang the whole app indefinitely.
        # Note: don't use the executor as a context manager here — `with` blocks
        # on shutdown() waiting for the worker thread to finish, which defeats
        # the timeout if the underlying network call itself never times out.
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = ex.submit(_load)
        try:
            return future.result(timeout=60)
        finally:
            ex.shutdown(wait=False)

    try:
        distilbert_tokenizer, distilbert_model = load_distilbert()
    except Exception as e:
        DISTILBERT_AVAILABLE = False
        DISTILBERT_LOAD_ERROR = str(e)

# Real aspect analyzer + SHAP explainer both require the DistilBERT model.
# Import lazily and gracefully so the rest of the app still works if either
# is unavailable (e.g. model still uploading, or shap not installed yet).
ASPECT_ANALYZER_AVAILABLE = False
SHAP_AVAILABLE = False
aspect_analyzer = None
shap_explainer = None

if DISTILBERT_AVAILABLE:
    try:
        import aspect_analyzer as _aspect_analyzer
        aspect_analyzer = _aspect_analyzer
        ASPECT_ANALYZER_AVAILABLE = True
    except Exception as e:
        ASPECT_ANALYZER_AVAILABLE = False

    try:
        import shap_explainer as _shap_explainer
        shap_explainer = _shap_explainer
        SHAP_AVAILABLE = True
    except Exception as e:
        SHAP_AVAILABLE = False

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

def predict_tfidf(text):
    X = tfidf_vectorizer.transform([str(text)])
    probs = tfidf_model.predict_proba(X)[0]
    classes = list(tfidf_model.classes_)
    ordered = np.array([probs[classes.index(l)] for l in LABELS])
    return LABELS[int(np.argmax(ordered))], ordered

def predict_distilbert(text):
    import torch
    inputs = distilbert_tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = distilbert_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    # numeric index order assumed 0=negative,1=neutral,2=positive per training convention
    return LABELS[int(np.argmax(probs))], probs

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(f"<div class='masthead' style='border:none;padding:0;margin-bottom:20px;'>"
                f"<div class='kicker'>COURSE FEEDBACK INTELLIGENCE</div>"
                f"<h1 style='font-size:22px;margin:6px 0 0 0;'>Coursera Corpus</h1></div>",
                unsafe_allow_html=True)

    workspace = st.radio(
        "Workspace",
        ["Live Inference", "Model Comparison", "Aspect Analysis", "Corpus Overview"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1px;background:#33404C;margin:18px 0;'></div>", unsafe_allow_html=True)

    engine_options = ["TF-IDF + Logistic Regression"]
    if DISTILBERT_AVAILABLE:
        engine_options.append("Multilingual DistilBERT")
    engine = st.selectbox("Inference engine", engine_options)

    if not DISTILBERT_AVAILABLE:
        st.markdown(
            "<div class='pending-note'>Multilingual DistilBERT: training in progress.<br/>"
            "This engine activates automatically once the fine-tuned model is added.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1px;background:#33404C;margin:18px 0;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-family:JetBrains Mono;font-size:11.5px;color:{MUTED};line-height:1.7;'>"
        f"DATASET&nbsp;&nbsp;Coursera reviews<br/>"
        f"ROWS&nbsp;&nbsp;{metrics['dataset_size_after_cleaning']:,}<br/>"
        f"CLASSES&nbsp;&nbsp;negative / neutral / positive</div>",
        unsafe_allow_html=True,
    )

# =====================================================================
# MASTHEAD
# =====================================================================
st.markdown(f"""
<div class="masthead">
    <div class="kicker">CAIPP BATCH 01 &middot; APPLIED NLP</div>
    <h1>Course Feedback Intelligence</h1>
    <div class="dek">Multilingual sentiment analysis over {metrics['dataset_size_after_cleaning']:,} real Coursera
    course reviews, comparing a classical TF-IDF baseline against a fine-tuned multilingual transformer.</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# WORKSPACE: LIVE INFERENCE
# =====================================================================
if workspace == "Live Inference":

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>INPUT &mdash; SINGLE REVIEW</div>", unsafe_allow_html=True)

        sample_choice = None
        if sample_df is not None:
            options = ["Write my own..."] + list(sample_df["Review"].astype(str).head(12))
            sample_choice = st.selectbox("Load a real review from the corpus", options, label_visibility="collapsed")

        default_text = "" if sample_choice in (None, "Write my own...") else sample_choice
        review_text = st.text_area("Review text", value=default_text, height=160, label_visibility="collapsed",
                                    placeholder="Type or paste a course review in any language...")
        run = st.button("Run inference")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown("<div class='panel-hero'>", unsafe_allow_html=True)
        st.markdown(f"<div class='panel-title' style='color:{MUTED};border-color:{GLASS_BORDER};'>RESULT</div>",
                    unsafe_allow_html=True)

        if run and review_text.strip():
            if engine.startswith("Multilingual") and DISTILBERT_AVAILABLE:
                label, probs = predict_distilbert(review_text)
            else:
                label, probs = predict_tfidf(review_text)

            color = SENTIMENT_COLOR.get(label, TEXT)
            confidence = float(np.max(probs)) * 100

            st.markdown(f"<div class='sentiment-badge' style='color:{color};'><span>{label}</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='result-verdict' style='color:{color};'>{confidence:.1f}% confidence</div>",
                        unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=[float(p) * 100 for p in probs], y=LABELS, orientation="h",
                marker_color=[SENTIMENT_COLOR[l] for l in LABELS],
            ))
            fig.update_layout(
                height=180, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 100], showgrid=False, color=MUTED),
                yaxis=dict(color=TEXT),
                font=dict(family="JetBrains Mono", size=12, color=TEXT),
            )
            st.plotly_chart(fig, use_container_width=True)

            if not engine.startswith("Multilingual"):
                st.caption("Explainability (per-word SHAP) is available once the Multilingual DistilBERT "
                           "engine is active — it explains the transformer's decisions directly.")
            elif not SHAP_AVAILABLE:
                st.caption("SHAP explainability is not currently available (module failed to load).")
        else:
            st.markdown(f"<div style='color:{MUTED};font-family:JetBrains Mono;font-size:13px;'>"
                        f"Enter a review and run inference to see the verdict here.</div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if run and review_text.strip() and engine.startswith("Multilingual") and SHAP_AVAILABLE:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>WORD-LEVEL EXPLAINABILITY (REAL SHAP)</div>", unsafe_allow_html=True)
        with st.spinner("Computing SHAP values (this can take a moment on CPU)..."):
            try:
                shap_values = shap_explainer.explain_review(review_text, max_evals=200)
                tokens = list(shap_values.data[0])
                pred_idx = LABELS.index(label)
                token_shap = [float(v[pred_idx]) for v in shap_values.values[0]]
                shap_df = pd.DataFrame({"Token": tokens, "Impact": token_shap})
                shap_df = shap_df[shap_df["Token"].str.strip() != ""]
                fig = px.bar(
                    shap_df, x="Token", y="Impact",
                    color="Impact", color_continuous_scale=[[0, RUST], [0.5, BG_MID], [1, TEAL]],
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(family="JetBrains Mono", color=TEXT),
                                   margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Positive bars push the prediction toward the shown class; negative bars push away from it. "
                           "Computed directly on the transformer via SHAP's Partition explainer.")
            except Exception as e:
                st.caption(f"SHAP explanation could not be computed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# WORKSPACE: MODEL COMPARISON
# =====================================================================
elif workspace == "Model Comparison":
    report = metrics["classification_report"]

    metric_cells = "".join(
        f"<div class='metric-cell'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div></div>"
        for label, value in [
            ("ACCURACY", f"{report['accuracy']*100:.1f}%"),
            ("MACRO F1", f"{report['macro avg']['f1-score']*100:.1f}%"),
            ("WEIGHTED F1", f"{report['weighted avg']['f1-score']*100:.1f}%"),
            ("TEST ROWS", f"{int(report['macro avg']['support']):,}"),
        ]
    )
    st.markdown(f"<div class='metric-row'>{metric_cells}</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>PER-CLASS F1 &mdash; TF-IDF + LOGISTIC REGRESSION</div>",
                    unsafe_allow_html=True)
        f1_df = pd.DataFrame({
            "Class": LABELS,
            "F1": [report[l]["f1-score"] * 100 for l in LABELS],
        })
        fig = px.bar(f1_df, x="Class", y="F1", color="Class",
                     color_discrete_map=SENTIMENT_COLOR, range_y=[0, 100])
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="JetBrains Mono", color=TEXT),
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0, BG_MID], [1, TEAL]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="JetBrains Mono", color=TEXT),
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if DISTILBERT_AVAILABLE and distilbert_metrics:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        db_report_cells = "".join(
            f"<div class='metric-cell'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div></div>"
            for label, value in [
                ("DISTILBERT ACCURACY", f"{distilbert_metrics['test_results']['test_accuracy']*100:.1f}%"),
                ("DISTILBERT MACRO F1", f"{distilbert_metrics['test_results']['test_f1_macro']*100:.1f}%"),
                ("TF-IDF MACRO F1", f"{report['macro avg']['f1-score']*100:.1f}%"),
                ("IMPROVEMENT", f"{(distilbert_metrics['test_results']['test_f1_macro'] - report['macro avg']['f1-score'])*100:+.1f} pts"),
            ]
        )
        st.markdown(f"<div class='metric-row'>{db_report_cells}</div>", unsafe_allow_html=True)

        col_c, col_d = st.columns(2, gap="large")
        with col_c:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>PER-CLASS F1 &mdash; MULTILINGUAL DISTILBERT</div>",
                        unsafe_allow_html=True)
            pcm = distilbert_metrics["per_class_metrics"]
            f1_df2 = pd.DataFrame({
                "Class": [l.upper() for l in pcm.keys()],
                "F1": [v["f1"] * 100 for v in pcm.values()],
            })
            fig = px.bar(f1_df2, x="Class", y="F1", color="Class",
                         color_discrete_map=SENTIMENT_COLOR, range_y=[0, 100])
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="JetBrains Mono", color=TEXT),
                               margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_d:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>CONFUSION MATRIX &mdash; DISTILBERT</div>",
                        unsafe_allow_html=True)
            cm2 = np.array(distilbert_metrics["confusion_matrix"])
            cm2_df = pd.DataFrame(cm2, index=LABELS, columns=LABELS)
            fig = px.imshow(cm2_df, text_auto=True, color_continuous_scale=[[0, BG_MID], [1, TEAL]])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="JetBrains Mono", color=TEXT),
                               margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    elif DISTILBERT_AVAILABLE:
        st.markdown(
            "<div class='pending-note'>Model is loaded but distilbert_metrics.json was not found "
            "in the app folder \u2014 add it to see the full comparison.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='pending-note'>Multilingual DistilBERT is not currently reachable"
            f"{': ' + DISTILBERT_LOAD_ERROR if DISTILBERT_LOAD_ERROR else ''}.</div>",
            unsafe_allow_html=True,
        )

# =====================================================================
# WORKSPACE: ASPECT ANALYSIS
# =====================================================================
elif workspace == "Aspect Analysis":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>ASPECT-BASED SENTIMENT</div>", unsafe_allow_html=True)

    if ASPECT_ANALYZER_AVAILABLE:
        aspect_sample = None
        if sample_df is not None:
            options = ["Write my own..."] + list(sample_df["Review"].astype(str).head(12))
            aspect_sample = st.selectbox("Load a real review", options, label_visibility="collapsed")
        default_text = "" if aspect_sample in (None, "Write my own...") else aspect_sample
        aspect_text = st.text_area("Review to analyze", value=default_text, height=120,
                                    label_visibility="collapsed",
                                    placeholder="Type or paste a course review...")
        run_aspect = st.button("Analyze aspects")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_aspect and aspect_text.strip():
            with st.spinner("Analyzing per-aspect sentiment..."):
                result = aspect_analyzer.analyze_review(aspect_text, use_tfidf=True)

            overall = result["overall_sentiment"]
            color = SENTIMENT_COLOR.get(overall, BG_DEEP)
            st.markdown(
                f"<div class='panel-hero'><div class='panel-title' style='color:{MUTED};border-color:{GLASS_BORDER};'>"
                f"OVERALL SENTIMENT</div>"
                f"<div class='sentiment-badge' style='color:{color};'><span>{overall}</span></div></div>",
                unsafe_allow_html=True,
            )

            if result["aspects"]:
                for aspect_name, data in result["aspects"].items():
                    a_color = SENTIMENT_COLOR.get(data["sentiment"], BG_DEEP)
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='panel-title'>{aspect_name.upper()}</div>"
                        f"<span class='sentiment-badge' style='color:{a_color};'><span>{data['sentiment']}</span></span>"
                        f"&nbsp;&nbsp;<span style='color:{MUTED};font-family:JetBrains Mono;font-size:12px;'>"
                        f"{data['confidence']*100:.1f}% confidence (DistilBERT) &middot; "
                        f"TF-IDF says {data['tfidf_sentiment']} ({data['tfidf_confidence']*100:.1f}%)</span>",
                        unsafe_allow_html=True,
                    )
                    for line in data["evidence"]:
                        st.markdown(f"<div class='evidence-line'>&ldquo;{line}&rdquo;</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No specific course-quality aspect keywords were matched in this review.")
    else:
        st.markdown(
            "<div class='pending-note'>Aspect analyzer requires the Multilingual DistilBERT model. "
            "It will activate automatically once the model finishes loading from Hugging Face Hub.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# WORKSPACE: CORPUS OVERVIEW
# =====================================================================
elif workspace == "Corpus Overview":
    if sample_df is not None:
        full_df = pd.read_csv(os.path.join(BASE, "reviews_by_course.csv"))
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>SENTIMENT DISTRIBUTION &mdash; FULL CORPUS</div>",
                    unsafe_allow_html=True)
        sentiment = full_df["Label"].apply(lambda r: "NEGATIVE" if r <= 2 else ("NEUTRAL" if r == 3 else "POSITIVE"))
        counts = sentiment.value_counts().reindex(LABELS)
        fig = px.bar(x=LABELS, y=counts.values, color=LABELS, color_discrete_map=SENTIMENT_COLOR)
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="JetBrains Mono", color=TEXT),
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>TOP COURSES BY REVIEW VOLUME</div>", unsafe_allow_html=True)
        top_courses = full_df["CourseId"].value_counts().head(10)
        st.dataframe(top_courses.rename("Reviews"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
