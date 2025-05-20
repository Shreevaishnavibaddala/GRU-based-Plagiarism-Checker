import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import re
import time

model = tf.keras.models.load_model("best_gru_model.keras", compile=False)
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def vectorize(text):
    text = preprocess(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
    return padded

st.title("Plagiarism Checker")

source_text = st.text_area("Enter Source Text (Click CTRL+Enter after entering text)")
plagiarism_text = st.text_area("Enter Text to Check for Plagiarism (Click CTRL+Enter after entering text)")

if st.button("Check"):

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
