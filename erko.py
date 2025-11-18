# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# NLTK download
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# --------------------
# TEXT CLEANING
# --------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# --------------------
# TRAIN MODEL
# --------------------
def train_model(df, text_col='text', label_col='label'):
    st.info("Модель үйретіліп жатыр, күтіңіз...")

    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    X = df[text_col].values
    y = df[label_col].values

    # stratify тек барлық класстарда 2 ден көп элемент болғанда
    unique_counts = df[label_col].value_counts()

    if unique_counts.min() < 2:
        st.warning("Stratify қолданылмайды, себебі кейбір класста 2 жазба жоқ.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_tf, y_train)

    preds = model.predict(X_test_tf)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds, zero_division=0)

    # SAVE MODEL + VECTORIZER
    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

    return acc, f1, report

# --------------------
# LOAD MODEL
# --------------------
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = pickle.load(open(MODEL_PATH, "rb"))
        vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
        return model, vectorizer
    return None, None

# --------------------
# PREDICT
# --------------------
def predict(text, model, vectorizer):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()
    return pred, prob, clean

# --------------------
# UI
# --------------------
st.title("📰 Fake News Detector — Logistic Regression")

model, vectorizer = load_model()

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Мәтін тексеру")
    inp = st.text_area("Мәтін енгіз", height=200)

    if st.button("Анықтау"):
        if model is None:
            st.error("Алдымен модельді үйретіңіз!")
        else:
            pred, prob, clean = predict(inp, model, vectorizer)
            label = "FAKE (Жалған)" if pred == 1 else "REAL (Шын)"
            st.subheader(f"Нәтиже: {label}")
            st.write(f"Сенімділік: {prob*100:.2f}%")
            st.code(clean)

with col2:
    st.header("Модель тренировкасы")
    file = st.file_uploader("CSV жүктеу (text, label)", type="csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        if st.button("Train"):
            acc, f1, rep = train_model(df)
            st.success("Модель дайын!")
            st.write("Accuracy:", acc)
            st.write("F1:", f1)
            st.text(rep)
