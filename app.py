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
    page_icon="\u25A0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# DESIGN SYSTEM
# Ink + parchment academic-instrument aesthetic. See design_notes.md.
# =====================================================================
INK = "#12181F"
INK_PANEL = "#1B232C"
PARCHMENT = "#F6F3EC"
RULE = "#33404C"
SLATE = "#8B96A3"
RUST = "#B0483E"      # negative
OCHRE = "#C98A2E"     # neutral
TEAL = "#2F8F6B"      # positive

SENTIMENT_COLOR = {"NEGATIVE": RUST, "NEUTRAL": OCHRE, "POSITIVE": TEAL,
                    "negative": RUST, "neutral": OCHRE, "positive": TEAL}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}
.stApp {{
    background-color: {INK};
    color: {PARCHMENT};
}}
section[data-testid="stSidebar"] {{
    background-color: {INK_PANEL};
    border-right: 1px solid {RULE};
}}
h1, h2, h3 {{
    font-family: 'Fraunces', serif;
    font-weight: 500;
    color: {PARCHMENT};
    letter-spacing: -0.01em;
}}
.masthead {{
    border-bottom: 1px solid {RULE};
    padding-bottom: 18px;
    margin-bottom: 28px;
}}
.masthead .kicker {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: {SLATE};
    letter-spacing: 0.04em;
}}
.masthead h1 {{
    font-size: 40px;
    margin: 6px 0 4px 0;
}}
.masthead .dek {{
    color: {SLATE};
    font-size: 15px;
    max-width: 640px;
    line-height: 1.5;
}}
.panel {{
    background-color: {INK_PANEL};
    border: 1px solid {RULE};
    padding: 22px 24px;
    margin-bottom: 18px;
}}
.panel-parchment {{
    background-color: {PARCHMENT};
    color: {INK};
    border: 1px solid {RULE};
    padding: 26px 28px;
    margin-bottom: 18px;
}}
.panel-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: {SLATE};
    letter-spacing: 0.03em;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid {RULE};
}}
.metric-row {{
    display: flex;
    gap: 0;
    border: 1px solid {RULE};
    margin-bottom: 18px;
}}
.metric-cell {{
    flex: 1;
    padding: 18px 20px;
    border-right: 1px solid {RULE};
}}
.metric-cell:last-child {{ border-right: none; }}
.metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: {SLATE};
    margin-bottom: 6px;
}}
.metric-value {{
    font-family: 'Fraunces', serif;
    font-size: 30px;
    color: {PARCHMENT};
}}
.sentiment-badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.03em;
    padding: 4px 12px;
    border: 1px solid currentColor;
}}
.result-verdict {{
    font-family: 'Fraunces', serif;
    font-size: 34px;
    margin: 10px 0 4px 0;
}}
.evidence-line {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: {SLATE};
    border-left: 2px solid {RULE};
    padding-left: 10px;
    margin: 4px 0;
}}
.pending-note {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: {OCHRE};
    border: 1px dashed {OCHRE};
    padding: 12px 14px;
}}
div.stButton > button {{
    background-color: {PARCHMENT};
    color: {INK};
    border: 1px solid {PARCHMENT};
    border-radius: 0;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    padding: 10px 22px;
}}
div.stButton > button:hover {{
    background-color: {INK};
    color: {PARCHMENT};
    border: 1px solid {PARCHMENT};
}}
[data-testid="stTextArea"] textarea {{
    background-color: {INK};
    color: {PARCHMENT};
    border: 1px solid {RULE};
    border-radius: 0;
    font-family: 'IBM Plex Sans', sans-serif;
}}
hr {{ border-color: {RULE}; }}
div[role="radiogroup"] label span:first-child {{
    border-color: {SLATE} !important;
}}
div[role="radiogroup"] label[data-checked="true"] span:first-child,
div[role="radiogroup"] input:checked + div {{
    background-color: {TEAL} !important;
    border-color: {TEAL} !important;
}}
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
        f"<div style='font-family:IBM Plex Mono;font-size:11.5px;color:{SLATE};line-height:1.7;'>"
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
        st.markdown("<div class='panel-parchment'>", unsafe_allow_html=True)
        st.markdown(f"<div class='panel-title' style='color:{SLATE};border-color:{RULE};'>RESULT</div>",
                    unsafe_allow_html=True)

        if run and review_text.strip():
            if engine.startswith("Multilingual") and DISTILBERT_AVAILABLE:
                label, probs = predict_distilbert(review_text)
            else:
                label, probs = predict_tfidf(review_text)

            color = SENTIMENT_COLOR.get(label, INK)
            confidence = float(np.max(probs)) * 100

            st.markdown(f"<div class='sentiment-badge' style='color:{color};'>{label}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='result-verdict' style='color:{color};'>{confidence:.1f}% confidence</div>",
                        unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=[float(p) * 100 for p in probs], y=LABELS, orientation="h",
                marker_color=[SENTIMENT_COLOR[l] for l in LABELS],
            ))
            fig.update_layout(
                height=180, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor=PARCHMENT, plot_bgcolor=PARCHMENT,
                xaxis=dict(range=[0, 100], showgrid=False, color=INK),
                yaxis=dict(color=INK),
                font=dict(family="IBM Plex Mono", size=12, color=INK),
            )
            st.plotly_chart(fig, use_container_width=True)

            if not engine.startswith("Multilingual"):
                st.caption("Explainability (per-word SHAP) is available once the Multilingual DistilBERT "
                           "engine is active — it explains the transformer's decisions directly.")
            elif not SHAP_AVAILABLE:
                st.caption("SHAP explainability is not currently available (module failed to load).")
        else:
            st.markdown(f"<div style='color:{SLATE};font-family:IBM Plex Mono;font-size:13px;'>"
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
                    color="Impact", color_continuous_scale=[[0, RUST], [0.5, INK_PANEL], [1, TEAL]],
                )
                fig.update_layout(paper_bgcolor=INK_PANEL, plot_bgcolor=INK_PANEL,
                                   font=dict(family="IBM Plex Mono", color=PARCHMENT),
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
        fig.update_layout(showlegend=False, paper_bgcolor=INK_PANEL, plot_bgcolor=INK_PANEL,
                           font=dict(family="IBM Plex Mono", color=PARCHMENT),
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0, INK_PANEL], [1, TEAL]])
        fig.update_layout(paper_bgcolor=INK_PANEL, plot_bgcolor=INK_PANEL,
                           font=dict(family="IBM Plex Mono", color=PARCHMENT),
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
            fig.update_layout(showlegend=False, paper_bgcolor=INK_PANEL, plot_bgcolor=INK_PANEL,
                               font=dict(family="IBM Plex Mono", color=PARCHMENT),
                               margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_d:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>CONFUSION MATRIX &mdash; DISTILBERT</div>",
                        unsafe_allow_html=True)
            cm2 = np.array(distilbert_metrics["confusion_matrix"])
            cm2_df = pd.DataFrame(cm2, index=LABELS, columns=LABELS)
            fig = px.imshow(cm2_df, text_auto=True, color_continuous_scale=[[0, INK_PANEL], [1, TEAL]])
            fig.update_layout(paper_bgcolor=INK_PANEL, plot_bgcolor=INK_PANEL,
                               font=dict(family="IBM Plex Mono", color=PARCHMENT),
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
            color = SENTIMENT_COLOR.get(overall, INK)
            st.markdown(
                f"<div class='panel-parchment'><div class='panel-title' style='color:{SLATE};border-color:{RULE};'>"
                f"OVERALL SENTIMENT</div>"
                f"<div class='sentiment-badge' style='color:{color};'>{overall}</div></div>",
                unsafe_allow_html=True,
            )

            if result["aspects"]:
                for aspect_name, data in result["aspects"].items():
                    a_color = SENTIMENT_COLOR.get(data["sentiment"], INK)
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='panel-title'>{aspect_name.upper()}</div>"
                        f"<span class='sentiment-badge' style='color:{a_color};'>{data['sentiment']}</span>"
                        f"&nbsp;&nbsp;<span style='color:{SLATE};font-family:IBM Plex Mono;font-size:12px;'>"
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
        fig.update_layout(showlegend=False, paper_bgcolor=INK_PANEL, plot_bgcolor=INK_PANEL,
                           font=dict(family="IBM Plex Mono", color=PARCHMENT),
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>TOP COURSES BY REVIEW VOLUME</div>", unsafe_allow_html=True)
        top_courses = full_df["CourseId"].value_counts().head(10)
        st.dataframe(top_courses.rename("Reviews"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
