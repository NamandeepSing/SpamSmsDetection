import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string
import os

# Download only what is truly needed
nltk.download('stopwords')

# Load model and vectorizer safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))

ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    words = tokenizer.tokenize(text)

    y = []
    for word in words:
        if word not in stop_words:
            y.append(ps.stem(word))

    return " ".join(y)

# UI
st.title("SMS Spam Detection")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam (Ham)")
