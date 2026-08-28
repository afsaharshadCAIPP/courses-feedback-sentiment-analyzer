import pickle
import streamlit as st

from aspect_analyzer import extract_aspects

st.set_page_config(
    page_title="Customer Feedback Sentiment Analyzer", layout="centered"
)

st.title("📊 Customer Feedback & Aspect Sentiment Analyzer")
st.write(
    "Enter a course review or customer feedback below to analyze its sentiment and extract aspect-level breakdown."
)


@st.cache_resource
def load_models():
  with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
  with open("course_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
  return vectorizer, model


vectorizer, model = load_models()

user_review = st.text_area(
    "Customer Review / Feedback",
    placeholder="Type or paste review here...",
    height=150,
)

if st.button("Analyze Sentiment", type="primary"):
  if not user_review.strip():
    st.warning("Please enter review text before analyzing.")
  else:
    # Predict overall sentiment
    X = vectorizer.transform([user_review])
    sentiment = model.predict(X)[0]
    aspects = extract_aspects(user_review)

    st.markdown("---")
    st.subheader("Analysis Results")

    # Display overall sentiment with color coding
    if sentiment == "positive":
      st.success(f"**Overall Sentiment:** Positive 😊")
    elif sentiment == "negative":
      st.error(f"**Overall Sentiment:** Negative 😞")
    else:
      st.info(f"**Overall Sentiment:** Neutral 😐")

    # Display Aspect breakdown
    st.markdown("### Aspect-Level Breakdown")
    if aspects:
      for aspect, details in aspects.items():
        st.write(f"- **{aspect.capitalize()}**: `{details}`")
    else:
      st.write("No specific aspects detected in this text.")
