import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import re
import nltk
import spacy
import os
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

model = tf.keras.models.load_model('best_gru_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def preprocess(text):
    text = text.lower()  # Lowercase text
    text = contractions.fix(text)  # Expand contractions 
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation, digits, special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    words = word_tokenize(text)  # Tokenize
    words = [w for w in words if w not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(w) for w in words]  # Lemmatize
    return ' '.join(words)

def vectorize(text):
    processed_text = preprocess(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
    return padded

st.title("Plagiarism Checker")

source_text = st.text_area("Enter Source Text")
plagiarism_text = st.text_area("Enter Text to Check for Plagiarism")

if st.button("Check"):
    if not source_text.strip() or not plagiarism_text.strip():
        st.error("Both fields must be filled in.")
    else:
        src_vec = vectorize(source_text)
        plag_vec = vectorize(plagiarism_text)

        combined = np.hstack((src_vec, plag_vec))
        combined = combined.reshape((combined.shape[0], 1, combined.shape[1]))

        prediction = model.predict(combined)[0][0]

        st.write("Plagiarism Probability:", round(prediction * 100, 2), "%")

        if prediction > 0.5:
            st.error("Plagiarism Detected")
        else:
            st.success("No Plagiarism Detected")
