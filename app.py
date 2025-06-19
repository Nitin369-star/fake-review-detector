import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import joblib
import os

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Fake Review Detector", layout="centered")

# Load ML model and vectorizer
model_path = "tiny_fake_review_model.pkl"
vec_path = "tiny_vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vec_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
else:
    model = None
    vectorizer = None
    st.warning("🚫 ML model files not found. Please train your model first.")

st.title("🕵️‍♂️ AI-Powered Fake Review Detector")

# Core logic for scoring
def check_review(review_text, rating=None):
    fake_score = 0
    reasons = []

    words = review_text.lower().split()
    word_freq = {word: words.count(word) for word in set(words)}
    repeated_words = [word for word, count in word_freq.items() if count > 3]
    if repeated_words:
        fake_score += 1
        reasons.append(f"Repeated words: {', '.join(repeated_words)}")

    if len(review_text) < 20 or len(review_text) > 300:
        fake_score += 1
        reasons.append(f"Suspicious length: {len(review_text)} characters")

    superlatives = ['amazing', 'incredible', 'unbelievable', 'best', 'worst', 'always', 'never']
    used_supers = [word for word in superlatives if word in review_text.lower()]
    if len(used_supers) >= 2:
        fake_score += 1
        reasons.append(f"Superlatives: {', '.join(used_supers)}")

    if review_text.isupper() or review_text.count("😂") + review_text.count("🔥") + review_text.count("💯") > 3:
        fake_score += 1
        reasons.append("All caps or emoji overload")

    if rating is not None:
        polarity = TextBlob(review_text).sentiment.polarity
        if rating <= 2 and polarity > 0.5:
            fake_score += 1
            reasons.append("Positive tone but low rating")

    score = max(0, 100 - (fake_score * 20))
    return score, fake_score, reasons

# Tabs for Single, Bulk, and ML prediction
tab1, tab2, tab3 = st.tabs([
    "📝 Check Single Review",
    "📂 Bulk Review CSV Upload",
    "🤖 ML-Based Prediction"
])

# --- Tab 1: Single Review ---
with tab1:
    review = st.text_area("✍️ Enter your product review")
    rating = st.slider("🌟 Star Rating", 1, 5, 3)

    if st.button("🔍 Analyze"):
        score, flags, reasons = check_review(review, rating)

        if flags >= 3:
            st.error("🚩 This review is likely **FAKE** ❌")
        elif flags == 2:
            st.warning("⚠️ This review is **Suspicious** 🤔")
        else:
            st.success("✅ This review looks **Genuine** 👍")

        st.metric("🧠 Genuineness Score", f"{score}%", delta=f"-{flags*20}%")
        st.progress(score / 100)

        for reason in reasons:
            st.write("📌", reason)

        labels = ['Repetition', 'Length', 'Superlatives', 'Spam']
        values = [
            int(any("Repeated" in r for r in reasons)),
            int(any("length" in r for r in reasons)),
            int(any("Superlatives" in r for r in reasons)),
            int(any("caps" in r or "emoji" in r for r in reasons))
        ]
        if any(values):
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.0f%%')
            st.pyplot(fig)

# --- Tab 2: Bulk Review ---
with tab2:
    uploaded_file = st.file_uploader("📂 Upload CSV with a 'review' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("❌ CSV must contain a 'review' column.")
        else:
            results = []
            for review in df["review"]:
                score, fake_score, reasons = check_review(str(review))
                if fake_score >= 3:
                    label = "Fake"
                elif fake_score == 2:
                    label = "Suspicious"
                else:
                    label = "Genuine"
                results.append({
                    "Review": review,
                    "Score (%)": score,
                    "Label": label,
                    "Reasons": ", ".join(reasons)
                })

            result_df = pd.DataFrame(results)
            st.dataframe(result_df)

            # PIE CHART
            label_counts = result_df["Label"].value_counts()
            if not label_counts.empty:
                fig, ax = plt.subplots()
                ax.pie(label_counts, labels=label_counts.index, autopct='%1.0f%%', startangle=90)
                st.subheader("📊 Review Classification Breakdown")
                st.pyplot(fig)

            # DOWNLOAD
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "review_results.csv", "text/csv")

# --- Tab 3: ML-Based Review Prediction ---
with tab3:
    st.header("🧠 ML-Based Review Classifier")
    ml_input = st.text_area("Enter a review to predict using ML model:")

    if st.button("ML Predict"):
        if model is not None and vectorizer is not None:
            vec = vectorizer.transform([ml_input])
            prediction = model.predict(vec)[0]
            st.success(f"Prediction: **{prediction}**")
        else:
            st.error("🚫 Model not loaded. Make sure .pkl files are in the app folder.")
