import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model
model = tf.keras.models.load_model("bilstm_fake_review_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("Fake Review Detector (Colab)")

review = st.text_area("Enter a review:")

if st.button("Check Review"):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=200)  # same maxlen as training
    pred = model.predict(padded)[0][0]
    if pred > 0.5:
        st.error(f"⚠️ Fake Review ({pred:.2f})")
    else:
        st.success(f"✅ Real Review ({pred:.2f})")
