import streamlit as st
st.set_page_config(page_title="Voice Review Detector", layout="centered")

import whisper
from tempfile import NamedTemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from deep_translator import GoogleTranslator
import joblib
import os

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

whisper_model = load_whisper_model()

# Load ML model and vectorizer
@st.cache_resource
def load_ml_model():
    model_path = "tiny_fake_review_model.pkl"
    vec_path = "tiny_vectorizer.pkl"
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return joblib.load(model_path), joblib.load(vec_path)
    else:
        return None, None

ml_model, vectorizer = load_ml_model()

st.title("📝 Voice Review Sentiment & Fake Review Detector")

# --- Audio file uploader and transcription ---
audio_file = st.file_uploader("🎙️ Upload or record your review", type=["mp3", "wav", "webm"])
if audio_file is not None:
    st.audio(audio_file)
    file_ext = os.path.splitext(audio_file.name)[1]
    with NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp.flush()
        try:
            with st.spinner("Transcribing audio..."):
                result = whisper_model.transcribe(tmp.name)
            st.success("Transcription complete!")
            st.write("🗣️ Transcribed:", result["text"])
        except Exception as e:
            st.error(f"Error during transcription: {e}")

# --- Translation function ---
def translate_review(text, target_lang='en'):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text  # fallback if translation fails

# --- Fake review detection logic ---
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
        # Case 1: Rating is low, but sentiment is very positive
        if rating <= 2 and polarity > 0.5:
            fake_score += 1
            reasons.append("Positive tone but low rating")
        # Case 2: Rating is high, but sentiment is very negative
        elif rating >= 4 and polarity < -0.3:
            fake_score += 1
            reasons.append("Negative tone but high rating")

    score = max(0, 100 - (fake_score * 20))
    return score, fake_score, reasons

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📝 Check Single Review",
    "📂 Bulk Review CSV Upload",
    "🤖 ML-Based Prediction"
])

# --- Tab 1: Single Review ---
with tab1:
    review = st.text_area("✍️ Enter your product review")
    from streamlit_mic_recorder import mic_recorder
    st.subheader("🎙️ Or use voice input:")
    audio_dict = mic_recorder(start_prompt="🎤 Record Review", stop_prompt="🛑 Stop", just_once=True, format="webm")

    # If user records audio, transcribe it
    if audio_dict and "bytes" in audio_dict:
        with NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_dict["bytes"])
            tmp.flush()
            try:
                with st.spinner("Transcribing voice input..."):
                    result = whisper_model.transcribe(tmp.name)
                review = result["text"]
                st.info(f"🗣️ Transcribed: {review}")
            except Exception as e:
                st.error(f"Error during voice transcription: {e}")

    rating = st.slider("🌟 Star Rating", 1, 5, 3)

    if st.button("🔍 Analyze"):
        translated = translate_review(review)
        score, flags, reasons = check_review(translated, rating)
        st.caption(f"🌐 Translated Review: {translated}")

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
                translated = translate_review(str(review))
                score, fake_score, reasons = check_review(translated)
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

            label_counts = result_df["Label"].value_counts()
            if not label_counts.empty:
                fig, ax = plt.subplots()
                ax.pie(label_counts, labels=label_counts.index, autopct='%1.0f%%', startangle=90)
                st.subheader("📊 Review Classification Breakdown")
                st.pyplot(fig)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "review_results.csv", "text/csv")

# --- Tab 3: ML-Based Prediction ---
with tab3:
    st.header("🧠 ML-Based Review Classifier")
    ml_input = st.text_area("Enter a review to predict using ML model:")

    if st.button("ML Predict"):
        translated = translate_review(ml_input)
        st.caption(f"🌐 Translated Review: {translated}")
        if ml_model is not None and vectorizer is not None:
            vec = vectorizer.transform([translated])
            prediction = ml_model.predict(vec)[0]
            st.success(f"Prediction: **{prediction}**")
        else:
            st.error("🚫 Model not loaded. Make sure .pkl files are in the app folder.")
