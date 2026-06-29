import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.caption("Machine Learning & NLP")

@st.cache_resource
def load_models():
    with open("tfidf.pkl","rb") as f:
        tfidf=pickle.load(f)
    with open("nb_model.pkl","rb") as f:
        nb=pickle.load(f)
    with open("lr_model.pkl","rb") as f:
        lr=pickle.load(f)
    with open("dt_model.pkl","rb") as f:
        dt=pickle.load(f)
    with open("rf_model.pkl","rb") as f:
        rf=pickle.load(f)
    return tfidf,nb,lr,dt,rf

tfidf,nb_model,lr_model,dt_model,rf_model=load_models()

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words("english"))

def clean_text(text):
    text=str(text).lower()
    cleaned=""
    for c in text:
        cleaned+=c if c.isalpha() or c==" " else " "
    words=cleaned.split()
    words=[lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

model_choice=st.selectbox(
    "🤖 Select ML Model",
    ["Logistic Regression","Naive Bayes","Random Forest","Decision Tree"]
)

accuracy={
    "Logistic Regression":"97.36%",
    "Naive Bayes":"92.15%",
    "Random Forest":"97.81%",
    "Decision Tree":"93.44%"
}

st.info(f"📊 Test Accuracy : {accuracy[model_choice]}")

news=st.text_area(
    "📝 Enter News Text",
    height=220,
    placeholder="Paste a news article or headline..."
)

models={
    "Logistic Regression":lr_model,
    "Naive Bayes":nb_model,
    "Random Forest":rf_model,
    "Decision Tree":dt_model
}

if st.button("🔍 Check News",use_container_width=True):

    if not news.strip():
        st.warning("Please enter some news text.")

    else:
        cleaned=clean_text(news)
        vector=tfidf.transform([cleaned])

        model=models[model_choice]

        pred=model.predict(vector)[0]
        prob=model.predict_proba(vector)[0]

        fake_prob=prob[0]*100
        real_prob=prob[1]*100
        confidence=max(fake_prob,real_prob)
st.divider()

        if pred == 1:
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")

        st.metric("Confidence", f"{confidence:.2f}%")

        if confidence >= 90:
            st.success("🟢 High Confidence")
        elif confidence >= 70:
            st.warning("🟡 Moderate Confidence")
        else:
            st.error("🔴 Low Confidence")
            st.warning("Prediction is uncertain. Verify using trusted news sources.")

        st.markdown("### 📊 Prediction Probability")

        st.write(f"🟢 Real News: {real_prob:.2f}%")
        st.progress(min(int(real_prob), 100))

        st.write(f"🔴 Fake News: {fake_prob:.2f}%")
        st.progress(min(int(fake_prob), 100))

st.divider()

st.caption(
    "⚠️ This model is trained on a publicly available fake news dataset. "
    "Predictions may not always be accurate for recent, regional, satirical, "
    "or AI-generated news. Please verify important news using trusted sources."
)
