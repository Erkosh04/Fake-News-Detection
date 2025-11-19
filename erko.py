# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ---------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ---------------------------------------------------
# FIND TEXT COLUMN
# ---------------------------------------------------
def find_text_column(df):
    for col in df.columns:
        if col.lower() in ["text", "content", "body", "article"]:
            return col
    return df.columns[0]   # fallback

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
def train_model(df):

    # Егер label жоқ болса → автомат түрде қосамыз
    if "label" not in df.columns:
        st.warning("⚠ Label баған жоқ → автомат түрде 'label = 1' қосылды.")
        df["label"] = 1

    text_col = find_text_column(df)
    label_col = "label"

    st.info(f"Text бағаны: **{text_col}**, Label бағаны: **{label_col}**")

    df[text_col] = df[text_col].astype(str).apply(clean_text)
    X = df[text_col].values
    y = df[label_col].values

    # Stratify fix
    unique, counts = np.unique(y, return_counts=True)
    if all(c >= 2 for c in counts):
        stratify = y
    else:
        stratify = None
        st.warning("⚠️ Stratify қолданылмайды — кейбір класс тек 1 дана ғана бар.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_tf, y_train)

    preds = model.predict(X_test_tf)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    rep = classification_report(y_test, preds)

    return model, vectorizer, acc, f1, rep

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.title("📰 Fake News Detection App")

uploaded = st.file_uploader("CSV файлын жүкте")

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Файл оқылды — {df.shape[0]} жол")

    if "label" not in df.columns:
        st.warning("⚠ CSV ішінде label жоқ → автомат түрде 1 қойылады (FAKE).")

    st.dataframe(df.head())
else:
    df = None

if st.button("Модельді үйрету"):
    if df is None:
        st.error("Алдымен CSV жүкте!")
    else:
        model, vectorizer, acc, f1, rep = train_model(df)
        st.success("Модель дайын!")

        st.write("🔹 **Accuracy:**", acc)
        st.write("🔹 **F1-score:**", f1)
        st.text(rep)

        # SAVE MODEL
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        st.info("Модель сақталды!")

st.markdown("---")

st.header("Мәтін тексеру")

text_input = st.text_area("Мәтінді енгіз:")

if st.button("Тексеру"):
    if not os.path.exists(MODEL_PATH):
        st.error("Алдымен модельді үйрет!")
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        clean = clean_text(text_input)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]

        label = "FAKE ❌" if str(pred) in ["1", "true", "True"] else "REAL ✔"

        st.subheader(label)
        st.code(clean)
