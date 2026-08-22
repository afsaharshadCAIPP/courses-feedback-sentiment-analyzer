import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
from aspect_analyzer import analyze_review_aspects, analyze_batch, ASPECT_KEYWORDS

@st.cache_data
def load_metrics():
    try:
        with open("metrics.json") as f:
            return json.load(f)
    except Exception:
        return None

# ==========================================
# 1. PAGE CONFIG & HIGH-CONTRAST DARK THEME
# ==========================================
st.set_page_config(
    page_title="Customer Feedback Sentiment Analyzer | Afsah Arshad",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS Fixes
css_code = """
<style>
    /* App Background */
    .stApp {
        background-color: #0b0f17;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Global Text & Label Fixes */
    label, .stWidgetLabel, div[data-testid="stMarkdownContainer"] p, h1, h2, h3, h4, h5, h6 {
        color: #f0f6fc !important;
    }
    
    /* Header Container & Developer Badge */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 22px 30px;
        background: #161b22;
        border: 1.5px solid #30363d;
        border-radius: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 25px rgba(0, 242, 254, 0.15);
    }
    
    .hero-title {
        background: linear-gradient(135deg, #00F2FE 0%, #38EF7D 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.2;
        filter: drop-shadow(0px 2px 12px rgba(0, 242, 254, 0.4));
    }
    .hero-sub {
        color: #8b949e !important;
        font-size: 1rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .dev-badge {
        background: #0d1117;
        border: 1.5px solid #00F2FE;
        padding: 12px 22px;
        border-radius: 14px;
        text-align: right;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.3);
        min-width: 200px;
    }
    .dev-label {
        color: #8b949e !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .dev-name {
        color: #00F2FE !important;
        font-weight: 800;
        font-size: 1.2rem;
    }

    /* Category Badges Bar */
    .category-bar {
        display: flex;
        gap: 12px;
        margin-bottom: 25px;
        justify-content: space-between;
    }
    .cat-card {
        flex: 1;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .cat-card:hover {
        border-color: #00F2FE;
        transform: translateY(-2px);
    }
    .cat-icon {
        font-size: 1.4rem;
        margin-bottom: 4px;
    }
    .cat-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #f0f6fc;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #8b949e !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0 20px !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f293d !important;
        color: #00F2FE !important;
        border: 1px solid #00F2FE !important;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.25) !important;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Badges */
    .badge-pos {
        background: rgba(16, 185, 129, 0.15);
        border: 1.5px solid #10B981;
        color: #10B981 !important;
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-block;
    }
    .badge-neg {
        background: rgba(239, 68, 68, 0.15);
        border: 1.5px solid #EF4444;
        color: #EF4444 !important;
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-block;
    }
    .badge-neu {
        background: rgba(245, 158, 11, 0.15);
        border: 1.5px solid #F59E0B;
        color: #F59E0B !important;
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-block;
    }

    /* Text Area */
    .stTextArea textarea {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: #0b0f17 !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 800 !important;
        border: none !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    .sidebar-dev-card {
        text-align: center;
        padding: 14px;
        background: #0d1117;
        border-radius: 12px;
        border: 1px solid #00F2FE;
        margin-bottom: 20px;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 20px;
        color: #8b949e !important;
        font-size: 0.85rem;
        border-top: 1px solid #30363d;
        margin-top: 40px;
    }
</style>
"""
st.markdown(css_code, unsafe_allow_html=True)

# Main Top Header Banner
header_html = """
<div class="header-container">
    <div>
        <p class="hero-title">⚡ Customer Feedback Sentiment Analyzer</p>
        <p class="hero-sub">Multi-Engine Sentiment Intelligence & Explainable AI Platform</p>
    </div>
    <div class="dev-badge">
        <div class="dev-label">Designed & Built By</div>
        <div class="dev-name">Afsah Arshad</div>
    </div>
</div>

<div class="category-bar">
    <div class="cat-card">
        <div class="cat-icon">👗</div>
        <div class="cat-title">Dresses & Gowns</div>
    </div>
    <div class="cat-card">
        <div class="cat-icon">👖</div>
        <div class="cat-title">Denim & Pants</div>
    </div>
    <div class="cat-card">
        <div class="cat-icon">👚</div>
        <div class="cat-title">Tops & Shirts</div>
    </div>
    <div class="cat-card">
        <div class="cat-icon">🧥</div>
        <div class="cat-title">Jackets & Coats</div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADERS
# ==========================================
@st.cache_resource
def load_tfidf():
    try:
        return joblib.load('clothing_sentiment_model.pkl'), joblib.load('tfidf_vectorizer.pkl')
    except Exception:
        return None, None

@st.cache_resource
def load_distilbert():
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", top_k=None)
    except Exception:
        return None

tfidf_model, vectorizer = load_tfidf()
distilbert_pipe = load_distilbert()
metrics_data = load_metrics()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
sidebar_dev_html = """
<div class="sidebar-dev-card">
    <div style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase;">DEVELOPER PROFILE</div>
    <div style="color: #ffffff; font-weight: 800; font-size: 1.15rem; margin-top:2px;">Afsah Arshad</div>
    <div style="color: #00F2FE; font-size: 0.85rem; font-weight: 600;">AI Practitioner & Specialist</div>
</div>
"""
st.sidebar.markdown(sidebar_dev_html, unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Engine Settings")
selected_model = st.sidebar.radio(
    "Select AI Model:",
    ["TF-IDF + Logistic Regression (ML)", "DistilBERT Transformer (Deep Learning)", "Ensemble (Compare Both)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Test Scenarios")
sample_choice = st.sidebar.selectbox(
    "Load Preset Feedback:",
    [
        "Custom Input",
        "The silk dress fabric is super soft, elegant, and fits perfectly!",
        "Jeans size is wrong, bad stitching, and color faded on wash.",
        "Decent top for daily wear, average fabric quality."
    ]
)

# ==========================================
# 4. MAIN TABS WORKSPACE
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Workspace", 
    "🔍 Explainable AI (XAI)", 
    "📊 Batch CSV Analytics", 
    "⚔️ Architecture Benchmarks",
    "🧩 Aspect-Based Insights",
])

# ------------------------------------------
# TAB 1: LIVE WORKSPACE
# ------------------------------------------
with tab1:
    default_text = "" if sample_choice == "Custom Input" else sample_choice
    user_input = st.text_area("📝 Enter Customer Feedback:", value=default_text, height=120)
    
    if st.button("🚀 Run Sentiment Intelligence", use_container_width=True):
        if user_input.strip():
            col1, col2 = st.columns([1, 1])
            
            pred_tfidf, probs_tfidf = "NEUTRAL", [0.33, 0.34, 0.33]
            pred_db, probs_db = "NEUTRAL", [0.33, 0.34, 0.33]

            if ("TF-IDF" in selected_model or "Compare" in selected_model) and vectorizer and tfidf_model:
                vec = vectorizer.transform([user_input])
                pred_tfidf = "NEUTRAL" if vec.nnz == 0 else tfidf_model.predict(vec)[0]
                probs_tfidf = [0.33, 0.34, 0.33] if vec.nnz == 0 else tfidf_model.predict_proba(vec)[0]

            if ("DistilBERT" in selected_model or "Compare" in selected_model) and distilbert_pipe:
                db_raw = distilbert_pipe(user_input)
                res_list = db_raw[0] if isinstance(db_raw, list) and isinstance(db_raw[0], list) else db_raw
                db_scores = {item['label'].upper(): item['score'] for item in res_list if isinstance(item, dict)}
                
                pos_score = db_scores.get('POSITIVE', 0.0)
                neg_score = db_scores.get('NEGATIVE', 0.0)
                neu_score = round(max(0.0, 1.0 - (pos_score + neg_score)), 4)
                
                pred_db = "POSITIVE" if pos_score > neg_score and pos_score > 0.5 else ("NEGATIVE" if neg_score > pos_score and neg_score > 0.5 else "NEUTRAL")
                probs_db = [neg_score, neu_score, pos_score]

            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Predicted Classification")
                final_pred = pred_db if "DistilBERT" in selected_model else pred_tfidf
                
                if final_pred == "POSITIVE":
                    st.markdown('<div class="badge-pos">🟢 POSITIVE</div>', unsafe_allow_html=True)
                elif final_pred == "NEGATIVE":
                    st.markdown('<div class="badge-neg">🔴 NEGATIVE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-neu">🟡 NEUTRAL</div>', unsafe_allow_html=True)

                st.write("")
                st.caption(f"Active Engine: `{selected_model}`")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Confidence Breakdown")
                active_probs = probs_db if "DistilBERT" in selected_model else probs_tfidf
                df_chart = pd.DataFrame({'Sentiment': ['NEGATIVE', 'NEUTRAL', 'POSITIVE'], 'Probability': active_probs})
                
                fig = px.bar(
                    df_chart, x='Probability', y='Sentiment', orientation='h',
                    color='Sentiment',
                    color_discrete_map={'POSITIVE': '#10B981', 'NEGATIVE': '#EF4444', 'NEUTRAL': '#F59E0B'},
                    text_auto='.1%'
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=200,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: EXPLAINABLE AI
# ------------------------------------------
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Word-Level Feature Weight Impact")
    st.write("Understand which specific words directly drove the AI's prediction:")
    
    if user_input.strip() and tfidf_model and vectorizer:
        vec = vectorizer.transform([user_input])
        pred_class = tfidf_model.predict(vec)[0]
        class_idx = list(tfidf_model.classes_).index(pred_class)

        words = user_input.lower().split()
        impact_list = []
        for word in words:
            if word in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[word]
                impact_list.append((word, tfidf_model.coef_[class_idx][idx]))
            else:
                impact_list.append((word, 0.0))

        st.caption(f"Showing word impact toward the predicted class: **{pred_class}**")
        df_impact = pd.DataFrame(impact_list, columns=['Word', 'Impact_Weight'])
        fig_xai = px.bar(
            df_impact, x='Word', y='Impact_Weight', 
            color='Impact_Weight', 
            color_continuous_scale='RdYlGn',
            title=f"Model Word Coefficients Impact (toward {pred_class})"
        )
        fig_xai.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_xai, use_container_width=True)
    else:
        st.info("💡 Enter customer text in 'Live Workspace' tab to view feature weights.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: BATCH ANALYTICS
# ------------------------------------------
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Bulk Feedback Analytics")
    uploaded_file = st.file_uploader("Upload CSV containing 'Review Text' column:", type=['csv', 'xlsx'])
    if uploaded_file and vectorizer and tfidf_model:
        df_batch = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if 'Review Text' in df_batch.columns:
            vecs = vectorizer.transform(df_batch['Review Text'].fillna(''))
            df_batch['Predicted_Sentiment'] = tfidf_model.predict(vecs)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Reviews", len(df_batch))
            m2.metric("Positive %", f"{(df_batch['Predicted_Sentiment']=='POSITIVE').mean()*100:.1f}%")
            m3.metric("Negative %", f"{(df_batch['Predicted_Sentiment']=='NEGATIVE').mean()*100:.1f}%")
            m4.metric("Neutral %", f"{(df_batch['Predicted_Sentiment']=='NEUTRAL').mean()*100:.1f}%")
            
            fig_donut = px.pie(
                df_batch, names='Predicted_Sentiment', hole=0.5, color='Predicted_Sentiment',
                color_discrete_map={'POSITIVE': '#10B981', 'NEGATIVE': '#EF4444', 'NEUTRAL': '#F59E0B'},
                title="Batch Sentiment Breakdown"
            )
            fig_donut.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: ARCHITECTURE BENCHMARKS (MATRIX & F1-SCORE)
# ------------------------------------------
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚔️ Model Architecture & Evaluation Metrics")

    if metrics_data is None:
        st.warning("⚠️ metrics.json not found. Run train_model_real.py first to generate real evaluation metrics.")
    else:
        st.caption(f"📁 Dataset: {metrics_data.get('dataset', 'N/A')}")
        st.caption(f"🏷️ Labels derived via: {metrics_data.get('label_source', 'N/A')}")

        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.markdown("#### 📊 Confusion Matrix (Real Test Set)")
            cm_data = metrics_data["confusion_matrix"]
            labels = metrics_data["labels"]

            fig_cm = px.imshow(
                cm_data,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Sample Count"),
                aspect="auto"
            )
            fig_cm.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=320
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_m2:
            st.markdown("#### 🎯 Classification Performance Report")
            report = metrics_data["classification_report"]
            rows = []
            for lbl in metrics_data["labels"]:
                r = report[lbl]
                rows.append({
                    "Sentiment Class": lbl,
                    "Precision": round(r["precision"], 3),
                    "Recall": round(r["recall"], 3),
                    "F1-Score": round(r["f1-score"], 3),
                    "Support": int(r["support"]),
                })
            rows.append({
                "Sentiment Class": "Weighted Avg",
                "Precision": round(report["weighted avg"]["precision"], 3),
                "Recall": round(report["weighted avg"]["recall"], 3),
                "F1-Score": round(report["weighted avg"]["f1-score"], 3),
                "Support": int(report["weighted avg"]["support"]),
            })
            df_report = pd.DataFrame(rows)
            st.dataframe(df_report, use_container_width=True, hide_index=True)

            st.markdown(f"""
            > **Overall Accuracy: {metrics_data['accuracy']*100:.1f}%** on a held-out real test set.
            > NEUTRAL (3-star) reviews are hardest to classify — this matches real-world
            > sentiment analysis behavior, since 3-star reviews mix positive and negative language.
            """)

        st.info(metrics_data.get("note", ""))

    st.markdown("---")
    st.markdown("#### Model Feature Comparison")
    tfidf_acc = f"{metrics_data['accuracy']*100:.1f}% (measured, this dataset)" if metrics_data else "N/A"
    st.markdown(f"""
    | Metric / Feature | TF-IDF + Logistic Regression | DistilBERT Transformer |
    | :--- | :--- | :--- |
    | **Architecture** | Linear Feature Matrix | Deep Bidirectional Transformer |
    | **Accuracy** | **{tfidf_acc}** | ~91% *(publicly reported SST-2 benchmark, not measured on this dataset)* |
    | **Inference Speed** | **< 5ms (Ultra Fast)** | ~80ms (Contextual) |
    | **Explainability (XAI)** | Exact Linear Weights | Attention Heatmaps / SHAP |
    """)
    st.caption("Note: DistilBERT accuracy above is a binary (positive/negative) benchmark from its model card, "
               "not tested on our 3-class dataset — shown for architecture comparison only, not a direct apples-to-apples number.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 5: ASPECT-BASED INSIGHTS
# ------------------------------------------
with tab5:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧩 Aspect-Based Sentiment Analysis")
    st.write(
        "Most sentiment tools give ONE label per review. This breaks a review into "
        "parts and reports sentiment separately for **Fabric/Quality, Fit/Sizing, "
        "Price/Value, Delivery/Shipping, and Color/Appearance** — so you know exactly "
        "*what* customers liked or disliked, not just *whether* they were happy."
    )
    st.caption(
        "⚙️ How it works: the review is split into clauses (on commas, periods, "
        "'but', 'however', etc.), each clause is matched to an aspect by keyword, "
        "then scored using the same TF-IDF model as the other tabs. Delivery/Shipping "
        "uses a small keyword-rule override, because the base clothing dataset barely "
        "discusses shipping — so the ML model has weak signal there."
    )
    st.markdown("---")

    sub_tab1, sub_tab2 = st.tabs(["🔎 Analyze One Review", "📊 Aggregate Across Many Reviews"])

    # ---- Single review breakdown ----
    with sub_tab1:
        st.caption(f"Active Engine (from sidebar): `{selected_model}`")
        aspect_input = st.text_area(
            "Enter a customer review:",
            value="Fabric is great but delivery was very late and honestly a bit overpriced.",
            height=100,
            key="aspect_input",
        )
        if st.button("🧩 Break Down by Aspect", use_container_width=True):
            if aspect_input.strip():
                badge_map = {"POSITIVE": "badge-pos", "NEGATIVE": "badge-neg", "NEUTRAL": "badge-neu"}
                icon_map = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}

                results_by_engine = {}
                if ("TF-IDF" in selected_model or "Compare" in selected_model) and vectorizer and tfidf_model:
                    results_by_engine["TF-IDF"] = analyze_review_aspects(
                        aspect_input, tfidf_model, vectorizer, engine="tfidf"
                    )
                if ("DistilBERT" in selected_model or "Compare" in selected_model) and distilbert_pipe:
                    results_by_engine["DistilBERT"] = analyze_review_aspects(
                        aspect_input, tfidf_model, vectorizer, engine="distilbert", distilbert_pipe=distilbert_pipe
                    )

                if not results_by_engine:
                    st.warning("No engine available — check model files loaded correctly.")
                elif not any(results_by_engine.values()):
                    st.info("No known aspects (fabric, fit, price, delivery, color) were mentioned in this review.")
                else:
                    for engine_name, result in results_by_engine.items():
                        if not result:
                            continue
                        st.markdown(f"##### Engine: {engine_name}")
                        cols = st.columns(len(result))
                        for col, (aspect, (sentiment, clause)) in zip(cols, result.items()):
                            with col:
                                st.markdown(f"**{aspect}**")
                                st.markdown(
                                    f'<div class="{badge_map[sentiment]}">{icon_map[sentiment]} {sentiment}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(f'"{clause}"')
                        st.markdown("")

    # ---- Aggregate across many reviews ----
    with sub_tab2:
        st.write("See which aspects drive the most negative feedback across many reviews at once.")
        st.caption(
            "ℹ️ Aggregate mode always uses TF-IDF (not DistilBERT), regardless of the sidebar "
            "engine setting — running a deep learning model on thousands of clauses would be "
            "too slow for an interactive app. Use the DistilBERT engine in 'Analyze One Review' above."
        )
        sample_n = st.slider("Number of real dataset reviews to sample:", 100, 2000, 500, step=100)

        if st.button("📊 Run Aggregate Aspect Analysis", use_container_width=True):
            if tfidf_model and vectorizer:
                try:
                    df_full = pd.read_csv("womens_clothing_reviews.csv")
                    sample_reviews = (
                        df_full["Review Text"].dropna().sample(n=sample_n, random_state=42).tolist()
                    )
                    with st.spinner(f"Analyzing {sample_n} reviews..."):
                        rows = analyze_batch(sample_reviews, tfidf_model, vectorizer)

                    if not rows:
                        st.info("No aspects detected in this sample.")
                    else:
                        df_agg = pd.DataFrame(rows)
                        fig_agg = px.bar(
                            df_agg, x="Aspect", y="Count", color="Sentiment",
                            barmode="group",
                            color_discrete_map={'POSITIVE': '#10B981', 'NEGATIVE': '#EF4444', 'NEUTRAL': '#F59E0B'},
                            title=f"Aspect Sentiment Breakdown (sample of {sample_n} real reviews)",
                        )
                        fig_agg.update_layout(
                            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_agg, use_container_width=True)

                        # Actionable takeaway: which aspect has the highest negative share
                        neg = df_agg[df_agg["Sentiment"] == "NEGATIVE"].set_index("Aspect")["Count"]
                        total = df_agg.groupby("Aspect")["Count"].sum()
                        neg_share = (neg / total).fillna(0).sort_values(ascending=False)
                        if len(neg_share) > 0:
                            top_issue = neg_share.index[0]
                            st.markdown(
                                f"> 💡 **Actionable insight:** *{top_issue}* has the highest share of negative "
                                f"mentions ({neg_share.iloc[0]*100:.0f}%) among reviews that mention it — "
                                f"this is where a business should focus first."
                            )
                except FileNotFoundError:
                    st.warning("womens_clothing_reviews.csv not found in this folder.")
    st.markdown('</div>', unsafe_allow_html=True)

# Custom Footer
footer_html = """
<div class="custom-footer">
    Developed with ❤️ by <strong style="color: #00F2FE;">Afsah Arshad</strong> | AI Practitioner & Data Science Specialist
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
