# ============================================================
# SHAP EXPLAINABLE AI
# Course Feedback Sentiment Analysis
#
# Explains predictions made by the fine-tuned
# multilingual DistilBERT sentiment model.
#
# CPU-ONLY VERSION FOR VS CODE
# ============================================================

import os
import numpy as np
import torch
import shap

from transformers import (AutoTokenizer,AutoModelForSequenceClassification)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_DIR = "./coursera_multilingual_distilbert"

MAX_LENGTH = 128

LABEL_NAMES = ["NEGATIVE","NEUTRAL","POSITIVE"]

# Number of SHAP evaluations.
# Lower = faster, higher = more detailed.
MAX_EVALS = 300


# ============================================================
# 2. FORCE CPU
# ============================================================

DEVICE = torch.device("cpu")

print("\n" + "=" * 70)
print("SHAP EXPLAINABLE AI")
print("=" * 70)

print("Device:", DEVICE)
print("CUDA will NOT be used.")


# ============================================================
# 3. CHECK MODEL
# ============================================================

# Prefer a local copy of the model if present, otherwise load straight from
# the Hugging Face Hub repo (the fine-tuned model is hosted there since the
# weights file is too large for a normal GitHub push).
MODEL_SOURCE = MODEL_DIR if os.path.exists(MODEL_DIR) else "Afsah-2027/coursera-multilingual-distilbert"


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_SOURCE
)

print("Tokenizer loaded successfully.")


# ============================================================
# 5. LOAD DISTILBERT MODEL
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SOURCE
)

model.to(DEVICE)

model.eval()

print("Model loaded successfully.")
print("Model device:", DEVICE)


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict_proba(texts):

    """
    Receives a list of text strings.

    Returns probability for:

        column 0 = NEGATIVE
        column 1 = NEUTRAL
        column 2 = POSITIVE
    """

    # SHAP may sometimes send a single string
    if isinstance(texts, str):
        texts = [texts]

    texts = [
        str(text)
        for text in texts
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    return probabilities.cpu().numpy()


# ============================================================
# 7. NORMAL SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):

    """
    Normal DistilBERT prediction.
    """

    probabilities = predict_proba(
        [text]
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    return {

        "sentiment":
            LABEL_NAMES[prediction],

        "confidence":
            float(
                probabilities[prediction]
            ),

        "probabilities": {

            "NEGATIVE":
                float(probabilities[0]),

            "NEUTRAL":
                float(probabilities[1]),

            "POSITIVE":
                float(probabilities[2])
        }
    }


# ============================================================
# 8. CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP text explainer...")

# SHAP uses the tokenizer as a text masker.
masker = shap.maskers.Text(
    tokenizer
)

explainer = shap.Explainer(
    predict_proba,
    masker
)

print("SHAP explainer created successfully.")


# ============================================================
# 9. GENERATE SHAP EXPLANATION
# ============================================================

def explain_review(
    text,
    max_evals=MAX_EVALS
):

    """
    Generate SHAP explanation for a review.
    """

    text = str(text).strip()

    if not text:

        print(
            "ERROR: Empty review."
        )

        return None

    print("\n" + "=" * 70)
    print("GENERATING SHAP EXPLANATION")
    print("=" * 70)

    print("\nReview:")
    print(text)

    # --------------------------------------------------------
    # First get normal prediction
    # --------------------------------------------------------

    prediction = predict_sentiment(
        text
    )

    print("\nModel Prediction:")
    print(
        "Sentiment:",
        prediction["sentiment"]
    )

    print(
        "Confidence:",
        f"{prediction['confidence']:.4f}"
    )

    print("\nProbabilities:")

    for label, probability in prediction[
        "probabilities"
    ].items():

        print(
            f"  {label}: "
            f"{probability:.4f}"
        )

    # --------------------------------------------------------
    # Generate SHAP values
    # --------------------------------------------------------

    print("\nCalculating SHAP values...")
    print(
        "This may take some time because "
        "the model is running on CPU."
    )

    shap_values = explainer(
        [text],
        max_evals=max_evals
    )

    return shap_values


# ============================================================
# 10. PRINT TOKEN CONTRIBUTIONS
# ============================================================

def print_shap_values(
    shap_values,
    text
):

    """
    Print the words/tokens that contributed
    most strongly to the prediction.
    """

    if shap_values is None:

        return

    print("\n")
    print("=" * 70)
    print("SHAP TOKEN CONTRIBUTIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Get prediction
    # --------------------------------------------------------

    prediction = predict_sentiment(
        text
    )

    predicted_class = (
        LABEL_NAMES.index(
            prediction["sentiment"]
        )
    )

    print(
        "\nExplaining:",
        prediction["sentiment"]
    )

    print(
        "Confidence:",
        f"{prediction['confidence']:.4f}"
    )

    # --------------------------------------------------------
    # Extract tokens
    # --------------------------------------------------------

    tokens = shap_values.data[0]

    values = shap_values.values[0]

    # Depending on SHAP version,
    # values can have different dimensions.
    #
    # Usually:
    #
    #   tokens x classes
    #
    # We select the predicted class.

    if values.ndim == 2:

        class_values = values[
            :,
            predicted_class
        ]

    else:

        class_values = values

    # --------------------------------------------------------
    # Make token contribution list
    # --------------------------------------------------------

    contributions = []

    for token, value in zip(
        tokens,
        class_values
    ):

        contributions.append(
            (
                str(token),
                float(value)
            )
        )

    # Sort by absolute importance
    contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(
        "\nMost important tokens:"
    )

    print("-" * 70)

    for token, value in contributions[:20]:

        if value > 0:

            direction = (
                "SUPPORTS "
                + prediction["sentiment"]
            )

        elif value < 0:

            direction = (
                "OPPOSES "
                + prediction["sentiment"]
            )

        else:

            direction = "NEUTRAL"

        print(
            f"{token:<25} "
            f"{value:>10.6f}   "
            f"{direction}"
        )


# ============================================================
# 11. SAVE SHAP HTML VISUALIZATION
# ============================================================

def save_shap_html(
    shap_values,
    filename="shap_explanation.html"
):

    """
    Save an interactive SHAP visualization
    as an HTML file.
    """

    if shap_values is None:

        print(
            "No SHAP values available."
        )

        return

    print(
        "\nSaving SHAP visualization..."
    )

    try:

        html = shap.plots.text(
            shap_values,
            display=False
        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                str(html)
            )

        print(
            "SHAP visualization saved to:"
        )

        print(
            os.path.abspath(filename)
        )

    except Exception as e:

        print(
            "\nCould not save SHAP HTML "
            "using the text plot."
        )

        print(
            "Error:",
            e
        )


# ============================================================
# 12. GENERATE CUSTOM SHAP BAR CHART
# ============================================================

def save_shap_bar_chart(
    shap_values,
    text,
    filename="shap_bar.png"
):

    """
    Create a custom SHAP bar chart for text sentiment.

    Positive SHAP values:
        Support the predicted sentiment.

    Negative SHAP values:
        Oppose the predicted sentiment.
    """

    if shap_values is None:
        return

    try:

        import matplotlib.pyplot as plt

        # ----------------------------------------------------
        # Get model prediction
        # ----------------------------------------------------

        prediction = predict_sentiment(text)

        predicted_class = LABEL_NAMES.index(
            prediction["sentiment"]
        )

        # ----------------------------------------------------
        # Get SHAP tokens and values
        # ----------------------------------------------------

        tokens = shap_values.data[0]

        values = shap_values.values[0]

        # ----------------------------------------------------
        # Select predicted sentiment class
        # ----------------------------------------------------

        if values.ndim == 2:

            class_values = values[
                :,
                predicted_class
            ]

        else:

            class_values = values

        # ----------------------------------------------------
        # Create token/value pairs
        # ----------------------------------------------------

        contributions = []

        for token, value in zip(
            tokens,
            class_values
        ):

            token = str(token).strip()

            value = float(value)

            if token:

                contributions.append(
                    (
                        token,
                        value
                    )
                )

        # ----------------------------------------------------
        # Remove extremely small contributions
        # ----------------------------------------------------

        contributions = [
            item
            for item in contributions
            if abs(item[1]) > 0.0001
        ]

        if not contributions:

            print(
                "No significant SHAP values "
                "available for chart."
            )

            return

        # ----------------------------------------------------
        # Select top 15 by absolute importance
        # ----------------------------------------------------

        contributions.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        contributions = contributions[:15]

        # ----------------------------------------------------
        # Reverse order for horizontal chart
        # ----------------------------------------------------

        contributions.reverse()

        tokens = [
            item[0]
            for item in contributions
        ]

        values = [
            item[1]
            for item in contributions
        ]

        # ----------------------------------------------------
        # Create chart
        # ----------------------------------------------------

        plt.figure(
            figsize=(12, 8)
        )

        plt.barh(
            tokens,
            values
        )

        plt.axvline(
            x=0,
            linewidth=1
        )

        plt.xlabel(
            "SHAP Value"
        )

        plt.ylabel(
            "Words / Tokens"
        )

        plt.title(
            "SHAP Explanation - "
            + prediction["sentiment"]
            + " Sentiment"
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # Save chart
        # ----------------------------------------------------

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nSHAP bar chart saved to:"
        )

        print(
            os.path.abspath(filename)
        )

    except Exception as e:

        print(
            "\nCould not create SHAP bar chart."
        )

        print(
            "Error:",
            e
        )


# ============================================================
# 13. ANALYZE ONE REVIEW
# ============================================================

def analyze_review(
    review,
    max_evals=MAX_EVALS
):

    """
    Complete SHAP analysis pipeline.
    """

    print("\n")

    shap_values = explain_review(
        review,
        max_evals=max_evals
    )

    if shap_values is None:

        return None

    print_shap_values(
        shap_values,
        review
    )

    return shap_values


# ============================================================
# 14. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example review
    # --------------------------------------------------------

    test_review = (
        "The instructor explained the concepts "
        "very clearly. The course content was "
        "excellent and the assignments were "
        "useful, although some exercises were "
        "quite difficult."
    )

    # --------------------------------------------------------
    # Run SHAP
    # --------------------------------------------------------

    shap_values = analyze_review(
        test_review
    )

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    if shap_values is not None:

        save_shap_html(
            shap_values,
            "shap_explanation.html"
        )

        save_shap_bar_chart(
            shap_values,
            "shap_bar.png"
        )

    # ========================================================
    # INTERACTIVE MODE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INTERACTIVE SHAP MODE")
    print("=" * 70)

    print(
        "\nEnter a course review."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        try:

            review = input(
                "\nEnter review: "
            )

        except KeyboardInterrupt:

            print(
                "\n\nProgram stopped."
            )

            break

        if review.lower().strip() == "exit":

            print(
                "\nSHAP analyzer stopped."
            )

            break

        if not review.strip():

            print(
                "Please enter a review."
            )

            continue

        shap_values = analyze_review(
            review
        )

        if shap_values is not None:

            save_shap_html(
                shap_values,
                "shap_explanation.html"
            )

            save_shap_bar_chart(
                shap_values,
                "shap_bar.png"
            )