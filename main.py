import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, Dense , SimpleRNN
from tensorflow.keras.models import load_model, Sequential

## Mapping word indices back to words
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the model
model = load_model('imdb_rnn_model.h5')

## Helper function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 2) + 3, 9999) for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=500)
    
## Prediction Function
## Prediction Function

def predict_review(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)[0][0]
    return 'Positive' if prediction >= 0.5 else 'Negative', prediction


## Streamlit App
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")
user_input = st.text_area("Movie Review", height=200)

if st.button("Predict Sentiment"):
    if user_input:
        sentiment, confidence = predict_review(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}** (Confidence: {confidence:.2f})")
    else:
        st.write("Please enter a movie review to predict its sentiment.")