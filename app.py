import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline

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
st.markdown("""
<style>
    /* App Background */
    .stApp {
        background-color: #0b0f17;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Global Text & Label Fixes for Readability */
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
        margin-bottom: 25px;
        box-shadow: 0 4px 25px rgba(0, 242, 254, 0.15);
    }
    
    /* HIGH-IMPACT PROMINENT PROJECT TITLE */
    .hero-title {
        background: linear-gradient(135deg, #00F2FE 0%, #38EF7D 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.2;
        filter: drop-shadow(0px 2px 12px rgba(0, 242, 254, 0.4));
    }
    .hero-sub {
        color: #8b949e !important;
        font-size: 1.05rem;
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
        font-size: 1.25rem;
    }

    /* Streamlit Tabs High Visibility */
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

    /* Glassmorphic Content Cards */
    .glass-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Glowing Badges */
    .badge-pos {
        background: rgba(16, 185, 129, 0.15);
        border: 1.5px solid #10B981;
        color: #10B981 !important;
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
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
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.25);
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
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }
    
    /* Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FAC
