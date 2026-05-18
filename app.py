import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.markdown("### Using Machine Learning & NLP")
st.markdown("---")

# Load saved models
@st.cache_resource
def load_models():
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('nb_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('dt_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    return tfidf, nb_model, lr_model, dt_model, rf_model

tfidf, nb_model, lr_model, dt_model, rf_model = load_models()

# EXACT same cleaning function as Jupyter notebook
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    cleaned = ""
    for char in text:
        if char.isalpha() or char == " ":
            cleaned += char
        else:
            cleaned += " "
    text = " ".join(cleaned.split())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words]
    return " ".join(words)

st.success("✅ Models loaded successfully!")
st.markdown("---")

# Model selector
model_choice = st.selectbox(
    "🤖 Select ML Model:",
    ['Logistic Regression', 'Naive Bayes',
     'Random Forest', 'Decision Tree']
)

accuracy_map = {
    'Logistic Regression': '97.36%',
    'Naive Bayes': '92.15%',
    'Random Forest': '97.85%',
    'Decision Tree': '93.10%'
}
st.info(f"📊 {model_choice} Accuracy: {accuracy_map[model_choice]}")

# Input
st.markdown("### 📝 Enter News Text Below:")
news_input = st.text_area(
    "",
    placeholder="Paste any news article or headline here...",
    height=200
)

if st.button("🔍 Check News", use_container_width=True):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text!")
    else:
        cleaned = clean_text(news_input)
        vectorized = tfidf.transform([cleaned])

        models_dict = {
            'Logistic Regression': lr_model,
            'Naive Bayes': nb_model,
            'Random Forest': rf_model,
            'Decision Tree': dt_model
        }

        selected_model = models_dict[model_choice]
        prediction = selected_model.predict(vectorized)[0]
        probability = selected_model.predict_proba(vectorized)[0]
        confidence = max(probability) * 100

        st.markdown("---")
        if prediction == 0:
            st.error("🔴 FAKE NEWS DETECTED!")
            st.markdown("### This news appears to be FAKE")
        else:
            st.success("🟢 REAL NEWS DETECTED!")
            st.markdown("### This news appears to be REAL")

        st.markdown(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))

st.markdown("---")
st.markdown("*Fake News Detection | 8th Sem Research Project*")