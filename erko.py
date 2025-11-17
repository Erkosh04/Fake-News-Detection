# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Ensure NLTK resources (first run will download)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

MODEL_PATH = "fake_news_lr_joblib.pkl"
VECTORIZER_PATH = "tfidf_vectorizer_joblib.pkl"

# ----------------------------
# Text preprocessing
# ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs, emails, special chars
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ----------------------------
# Train model pipeline
# ----------------------------
def train_model(df: pd.DataFrame, text_col='text', label_col='label', test_size=0.2, random_state=42, max_features=5000):
    st.info("Модель үйретіліп жатыр — күте тұрыңыз...")
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    X = df[text_col].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='binary') if len(np.unique(y))==2 else f1_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds, zero_division=0)

    # Save model & vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    return {
        "model": model,
        "vectorizer": tfidf,
        "accuracy": acc,
        "f1": f1,
        "report": report
    }

# ----------------------------
# Load saved pipeline
# ----------------------------
def load_saved_pipeline():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(VECTORIZER_PATH)
        return model, tfidf
    return None, None

# ----------------------------
# Predict helper
# ----------------------------
def predict_text(text: str, model, vectorizer):
    text_clean = clean_text(text)
    vect = vectorizer.transform([text_clean])
    prob = model.predict_proba(vect)[0] if hasattr(model, "predict_proba") else None
    pred = model.predict(vect)[0]
    # if binary, assume label 1 = fake or depending dataset; show prob for predicted class
    if prob is not None:
        # show probability for class '1' if labels are 0/1; otherwise show max probability
        prob_max = np.max(prob)
        return pred, float(prob_max)
    else:
        return pred, None

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")

st.title("📰 Fake News Detection — Streamlit App")
st.caption("Еркош братанға арналған жеңіл веб-қосымша. TF-IDF + Logistic Regression пайплайн.")

col1, col2 = st.columns([2,1])

with col1:
    st.header("1) Мақала мәтінін тексер")
    user_text = st.text_area("Мәтінді осында енгіз", height=200, placeholder="Мұнда жаңалық мәтінін қойыңыз...")
    model, vectorizer = load_saved_pipeline()
    if st.button("Талдау (Analyze)"):
        if model is None or vectorizer is None:
            st.warning("Алдымен модельді үйретіңіз немесе файлдан жүктеңіз (сол жақ).")
        else:
            pred, prob = predict_text(user_text, model, vectorizer)
            st.markdown("### Нәтиже:")
            # Try to interpret label meaning — we can't be sure label encoding; assume 1 = fake, 0 = real
            if isinstance(pred, (list, np.ndarray)):
                pred = pred[0]
            label_text = "FAKE (Жалған)" if str(pred) in ['1','True','true','yes'] else "REAL (Шын)"
            st.write(f"**Label:** {label_text}")
            if prob is not None:
                st.write(f"**Confidence:** {prob*100:.2f}%")
            # show cleaned text
            st.markdown("**Cleaned text (for model):**")
            st.code(clean_text(user_text))

with col2:
    st.header("2) Деректер және модель")
    st.markdown("**Жүктеу / Үйрету / Жүктеу (upload)**")
    uploaded_file = st.file_uploader("CSV жүктеу (столбецтер: 'text' және 'label')", type=['csv'])
    use_sample = st.checkbox("Мысал датасетті қолдану (егер жоқ болса)", value=False)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Файл оқылды — {df.shape[0]} жол, {df.shape[1]} баған")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error("CSV оқуда қате: " + str(e))
            df = None
    else:
        if use_sample:
            st.info("Мысал датасет жасалып жатыр (саңылаулармен).")
            # create a tiny synthetic example if real dataset not provided
            df = pd.DataFrame({
                "text": [
                    "Breaking: Scientists discover cure for disease",
                    "Celebrity scandal: shocking photos leaked",
                    "Local news: charity event helped families",
                    "Unbelievable! This pill burns fat overnight",
                    "Government announces new education reforms"
                ],
                "label": [0, 1, 0, 1, 0]  # 1 = fake, 0 = real (assumed)
            })
            st.dataframe(df)
        else:
            df = None

    # Train button
    st.markdown("---")
    if st.button("Үйрету (Train model)"):
        if df is None:
            st.error("Датасет жоқ — CSV файлын жүктеңіз немесе 'Мысал датасетті қолдану' қойыңыз.")
        else:
            # try to detect column names
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            label_col = 'label' if 'label' in df.columns else df.columns[1] if len(df.columns) > 1 else st.error("Label бағанын анықтаңыз.")
            res = train_model(df, text_col=text_col, label_col=label_col)
            st.success("Модель үйретілді және сақталды.")
            st.write(f"Accuracy: **{res['accuracy']:.4f}**")
            st.write(f"F1-score: **{res['f1']:.4f}**")
            st.text("Classification report:")
            st.text(res['report'])

    st.markdown("---")
    st.header("3) Модель файлдары")
    # Show if model exists
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        st.success("Saved model табылды.")
        st.write(f"- {MODEL_PATH}")
        st.write(f"- {VECTORIZER_PATH}")
        if st.button("Модельді жүктеу (Load model)"):
            st.info("Модель жүктелді.")
            # nothing else needed; functions load from disk when predicting
    else:
        st.info("Saved model табылмады. Үйретіп, сақтаңыз.")

    st.markdown("---")
    st.caption("Ескерту: label-дереккөздегі кодтау 0/1 түрінде екеніне көз жеткізіңіз (1 = fake, 0 = real) немесе өз датасетіңізге қарай кодтауды өзгертіңіз.")


