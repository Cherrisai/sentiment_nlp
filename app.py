from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import streamlit as st

from src.sentiment_analysis.config import DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_PATH, PROJECT_NAME
from src.sentiment_analysis.predict import load_model, predict_dataframe, predict_sentiment


st.set_page_config(page_title=PROJECT_NAME, layout="wide")


SENTIMENT_COLORS = {
    "Positive": "#11864b",
    "Neutral": "#b27800",
    "Negative": "#b42318",
}

SENTIMENT_EMOJIS = {
    "Positive": "Great",
    "Neutral": "Average",
    "Negative": "Concern",
}


@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data
def load_training_data() -> pd.DataFrame:
    if DEFAULT_PROCESSED_PATH.exists():
        return pd.read_csv(DEFAULT_PROCESSED_PATH)
    return pd.DataFrame()


def add_page_style() -> None:
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            color: #172033;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #526070;
            font-size: 1rem;
            margin-bottom: 1.1rem;
        }
        .animated-card {
            animation: fadeSlide 650ms ease-out;
            border: 1px solid #e3e8ef;
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(21, 32, 51, 0.08);
            margin: 0.7rem 0 1rem 0;
        }
        .sentiment-label {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
        }
        .answer-box {
            animation: fadeSlide 550ms ease-out;
            border-left: 5px solid #2457a6;
            background: #f5f7fb;
            padding: 1rem 1.1rem;
            border-radius: 8px;
            margin-top: 0.8rem;
        }
        .footer {
            color: #6b7280;
            text-align: center;
            padding: 2rem 0 0.4rem 0;
            font-size: 0.9rem;
        }
        @keyframes fadeSlide {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def detect_language_family(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "Indic language text detected"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "Telugu text detected"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "Tamil text detected"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "Kannada text detected"
    if re.search(r"[\u0600-\u06FF]", text):
        return "Arabic/Persian/Urdu text detected"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "Chinese/Japanese text detected"
    if re.search(r"[^\x00-\x7F]", text):
        return "Multilingual text detected"
    return "English or romanized text detected"


def render_sentiment_card(result: dict, heading: str = "Prediction Result") -> None:
    sentiment = result["sentiment"]
    color = SENTIMENT_COLORS.get(sentiment, "#172033")
    confidence = result.get("confidence")
    confidence_text = f"{confidence:.1%}" if confidence is not None else "Not available"
    st.markdown(
        f"""
        <div class="animated-card">
            <div style="font-size:0.9rem;color:#526070;font-weight:700;">{heading}</div>
            <p class="sentiment-label" style="color:{color};">{sentiment}</p>
            <div style="color:#526070;margin-top:0.25rem;">
                Signal: <b>{SENTIMENT_EMOJIS.get(sentiment, "Detected")}</b> |
                Confidence: <b>{confidence_text}</b>
            </div>
            <div style="color:#526070;margin-top:0.45rem;">
                Reason: <b>{result.get("reason", "text signal")}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "probabilities" in result:
        probability_df = (
            pd.Series(result["probabilities"], name="Probability")
            .rename_axis("Sentiment")
            .reset_index()
        )
        st.bar_chart(probability_df, x="Sentiment", y="Probability", color="#2457a6")


def render_batch_dashboard(predictions: pd.DataFrame) -> None:
    counts = predictions["predicted_sentiment"].value_counts()
    for label in ["Positive", "Negative", "Neutral"]:
        if label not in counts:
            counts.loc[label] = 0
    counts = counts[["Positive", "Negative", "Neutral"]]
    total = int(counts.sum())

    st.markdown("### Sentiment Dashboard")
    cols = st.columns(4)
    cols[0].metric("Total Reviews", total)
    cols[1].metric("Positive", int(counts["Positive"]))
    cols[2].metric("Negative", int(counts["Negative"]))
    cols[3].metric("Neutral", int(counts["Neutral"]))

    chart_df = counts.rename("Reviews").rename_axis("Sentiment").reset_index()
    st.bar_chart(chart_df, x="Sentiment", y="Reviews", color="#2457a6")

    percentage_df = (counts / total * 100).round(2).rename("Percentage").rename_axis("Sentiment").reset_index()
    st.dataframe(percentage_df, use_container_width=True, hide_index=True)


def filter_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = st.segmented_control(
        "Filter prediction results",
        options=["All", "Positive", "Negative", "Neutral"],
        default="All",
    )
    if selected == "All":
        return predictions
    return predictions[predictions["predicted_sentiment"] == selected]


def choose_text_column(df: pd.DataFrame) -> str:
    candidates = ["review_text", "body", "review", "reviews", "text", "comment", "title"]
    lowercase_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowercase_map:
            return lowercase_map[candidate]
    return df.columns[0]


def answer_question(question: str, model, predictions: pd.DataFrame | None, training_df: pd.DataFrame) -> str:
    question_clean = question.strip()
    question_lower = question_clean.lower()
    active_df = predictions if predictions is not None and not predictions.empty else training_df
    sentiment_column = "predicted_sentiment" if predictions is not None and not predictions.empty else "sentiment"

    if not question_clean:
        return "Please ask a question or enter a review."

    if any(word in question_lower for word in ["predict", "sentiment of", "review is", "classify", "analyse", "analyze"]):
        result = predict_sentiment(question_clean, model)
        confidence = result.get("confidence", 0)
        return (
            f"Prediction: {result['sentiment']} with {confidence:.1%} confidence. "
            f"Reason: {result.get('reason', 'review text signal')}."
        )

    if active_df.empty or sentiment_column not in active_df.columns:
        result = predict_sentiment(question_clean, model)
        confidence = result.get("confidence", 0)
        return f"I do not have dashboard data loaded yet, so I treated this as review text. Sentiment: {result['sentiment']} ({confidence:.1%})."

    counts = active_df[sentiment_column].value_counts()
    total = int(counts.sum())
    positive = int(counts.get("Positive", 0))
    negative = int(counts.get("Negative", 0))
    neutral = int(counts.get("Neutral", 0))

    if any(word in question_lower for word in ["how many", "count", "total"]):
        return f"There are {total} reviews: {positive} positive, {negative} negative, and {neutral} neutral."

    if any(word in question_lower for word in ["percentage", "percent", "%"]):
        if total == 0:
            return "There are no reviews available for percentage calculation."
        return (
            f"Positive: {positive / total:.1%}, "
            f"Negative: {negative / total:.1%}, "
            f"Neutral: {neutral / total:.1%}."
        )

    if any(word in question_lower for word in ["majority", "most", "highest"]):
        top = counts.idxmax()
        return f"The majority sentiment is {top} with {int(counts.max())} reviews."

    if any(word in question_lower for word in ["example", "sample", "show"]):
        for sentiment in ["Negative", "Positive", "Neutral"]:
            if sentiment.lower() in question_lower:
                subset = active_df[active_df[sentiment_column] == sentiment]
                if subset.empty:
                    return f"No {sentiment.lower()} review examples are available."
                text_col = "review_text_used" if "review_text_used" in subset.columns else "review_text"
                if text_col not in subset.columns:
                    text_col = "body" if "body" in subset.columns else subset.columns[0]
                sample = str(subset.iloc[0][text_col])[:350]
                return f"Example {sentiment.lower()} review: {sample}"

    if "negative" in question_lower:
        return f"Negative reviews: {negative} out of {total}."
    if "positive" in question_lower:
        return f"Positive reviews: {positive} out of {total}."
    if "neutral" in question_lower:
        return f"Neutral reviews: {neutral} out of {total}."

    if any(word in question_lower for word in ["common", "words", "keyword", "topic"]):
        text_columns = [column for column in ["review_text", "body", "title"] if column in active_df.columns]
        if "review_text_used" in active_df.columns:
            text_columns = ["review_text_used"]
        text = " ".join(active_df[text_columns].fillna("").astype(str).agg(" ".join, axis=1))
        tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
        stop_words = {"this", "that", "with", "have", "from", "phone", "mobile", "product", "very"}
        common = [word for word, _ in Counter(t for t in tokens if t not in stop_words).most_common(8)]
        return "Common review keywords: " + ", ".join(common) if common else "No common keywords found."

    if len(question_clean.split()) >= 3:
        result = predict_sentiment(question_clean, model)
        confidence = result.get("confidence", 0)
        return (
            f"This looks like review text. Prediction: {result['sentiment']} "
            f"with {confidence:.1%} confidence. Reason: {result.get('reason', 'text signal')}."
        )

    result = predict_sentiment(question_clean, model)
    confidence = result.get("confidence", 0)
    return f"Prediction: {result['sentiment']} with {confidence:.1%} confidence."


add_page_style()

st.markdown(f'<div class="main-title">{PROJECT_NAME}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Customer review sentiment analysis with single prediction, batch dashboard, and review Q&A.</div>',
    unsafe_allow_html=True,
)

try:
    model = get_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

training_df = load_training_data()
tab_single, tab_batch, tab_ask = st.tabs(["Single Review", "Batch Prediction", "Ask Questions"])

with tab_single:
    review_text = st.text_area(
        "Enter customer review in any language or romanized text",
        height=180,
        placeholder="Example: The phone battery is excellent, but the camera could be better.",
    )
    rating_option = st.selectbox(
        "Optional rating signal",
        options=["No rating", "1", "2", "3", "4", "5"],
        index=0,
        help="If rating is available, the prediction uses both rating and review words.",
    )

    if st.button("Predict Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review before prediction.")
        else:
            st.caption(detect_language_family(review_text))
            rating = None if rating_option == "No rating" else rating_option
            result = predict_sentiment(review_text, model, rating=rating)
            render_sentiment_card(result)

with tab_batch:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        st.write("Preview")
        st.dataframe(input_df.head(), use_container_width=True)

        default_column = choose_text_column(input_df)
        text_column = st.selectbox(
            "Select review text column",
            input_df.columns,
            index=list(input_df.columns).index(default_column),
            help="If title/body/rating columns are present, the app automatically uses them also for a stronger prediction.",
        )

        if st.button("Run Batch Prediction", type="primary"):
            predictions = predict_dataframe(input_df, text_column=text_column, model=model)
            st.session_state["batch_predictions"] = predictions
            st.success(f"Predicted {len(predictions)} reviews.")

    predictions = st.session_state.get("batch_predictions")
    if predictions is not None and not predictions.empty:
        render_batch_dashboard(predictions)
        filtered_predictions = filter_predictions(predictions)
        st.write(f"Prediction Results ({len(filtered_predictions)} rows)")
        st.dataframe(filtered_predictions, use_container_width=True)
        st.download_button(
            label="Download Predictions CSV",
            data=filtered_predictions.to_csv(index=False).encode("utf-8"),
            file_name="p676_sentiment_predictions.csv",
            mime="text/csv",
        )

with tab_ask:
    st.write("Ask about uploaded batch results, counts, percentages, examples, common topics, or paste a review for prediction.")
    suggestion = st.selectbox(
        "Suggestion box",
        options=[
            "Choose a question",
            "How many positive negative neutral reviews?",
            "What is the sentiment percentage?",
            "Which sentiment is highest?",
            "Show one negative review example",
            "Show one positive review example",
            "What are common review keywords?",
        ],
    )
    default_question = "" if suggestion == "Choose a question" else suggestion
    question = st.text_area(
        "Ask anything about sentiment results",
        value=default_question,
        height=110,
        placeholder="Example: how many positive negative neutral reviews?",
    )
    if st.button("Get Answer", type="primary"):
        answer = answer_question(
            question,
            model=model,
            predictions=st.session_state.get("batch_predictions"),
            training_df=training_df,
        )
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Copyrights to Saivignesh @ 2026</div>', unsafe_allow_html=True)
