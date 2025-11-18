# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import re

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
# FIND TEXT / LABEL COLUMNS
# ---------------------------------------------------
def find_text_column(df):
    for col in df.columns:
        if col.lower() in ["text", "content", "body", "article"]:
            return col
    return df.columns[0]   # fallback


def find_label_column(df):
    for col in df.columns:
        if col.lower() in ["label", "class", "fake", "target", "is_fake"]:
            return col
    return df.columns[-1]   # fallback


# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
def train_model(df):
    text_col = find_text_column(df)
    label_col = find_label_column(df)

    st.info(f"Text бағаны: **{text_col}**, Label бағаны: **{label_col}**")

    df[text_col] = df[text_col].astype(str).apply(clean_text)
    X = df[text_col].values
    y = df[label_col].values

    stratify = y if len(np.unique(y)) > 1 else None

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
st.title("📰 Fake News Detection App (Erkos Edition)")

uploaded = st.file_uploader("CSV файлын жүкте")

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Файл оқылды — {df.shape[0]} жол")
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

        # SAVE (NO joblib)
        import pickle
        pickle.dump(model, open(MODEL_PATH, "wb"))
        pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

        st.info("Модель сақталды!")

st.markdown("---")

st.header("Мәтін тексеру")

text_input = st.text_area("Мәтінді енгіз:")

if st.button("Тексеру"):
    if not os.path.exists(MODEL_PATH):
        st.error("Алдымен модельді үйрет!")
    else:
        import pickle
        model = pickle.load(open(MODEL_PATH, "rb"))
        vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

        clean = clean_text(text_input)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]

        label = "FAKE ❌" if str(pred) in ["1", "true", "True"] else "REAL ✔"

        st.subheader(label)
        st.code(clean)
