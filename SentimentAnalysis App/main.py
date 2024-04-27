import pandas as pd
import nltk
import time
import streamlit as st
import re

from nltk.stem.porter import *
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# link dataset: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'^[^ ]<.*?>|&([a-z0-9]+|#[0-9]\"\'\“{1,6}|#x[0-9a-f]{1,6});[^A-Za-z0-9]+')
    return re.sub(html, '', text)

def remove_quotes(text):
    quotes = re.compile(r'[^A-Za-z0-9\s]+')
    return re.sub(quotes, '', text)

def remove_punct(text):
    return re.sub(r'[^\w\s]', '', text)


def stem_words(text):
    stemmer = SnowballStemmer("english")
    return " ".join([stemmer.stem(word) for word in text.split()])


def preprocessing(sent):
    sent = remove_URL(sent)
    sent = remove_emoji(sent)
    sent = remove_html(sent)
    sent = remove_punct(sent)
    sent = remove_quotes(sent)
    sent = stem_words(sent)
    return sent

def prediction(s, vectorizer, model):
    bow = vectorizer.transform([s])
    predicted_sentiment = model.predict(bow)
    return predicted_sentiment


def main():
    #st.title("Sentiment Analysis Using NaiveBayes Model")
    st.markdown("<h1 style='font-size: 40px;'>Sentiment Analysis Using NaiveBayes</h1>", unsafe_allow_html=True)
    df = pd.read_csv("D:\VisualStudioCode\CS313\Tweets.csv")
    
    df = df.dropna()
    df['processed_text'] = df['selected_text'].apply(lambda x: remove_URL(x))
    df['processed_text'] = df['processed_text'].apply(lambda x: remove_emoji(x))
    df['processed_text'] = df['processed_text'].apply(lambda x: remove_html(x))
    df['processed_text'] = df['processed_text'].apply(lambda x: remove_punct(x))
    df['processed_text'] = df['processed_text'].apply(lambda x: remove_quotes(x))
    
    df['processed_text'] = df['processed_text'].apply(lambda x: stem_words(x))
    
    stop_words = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=word_tokenize)
    bow = vectorizer.fit_transform(df['processed_text'])
    
    label_encode = LabelEncoder()
    df['sentiment'] = label_encode.fit_transform(df['sentiment'])
    
    X = bow
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write('Completed training Naive Bayes model => Accuracy:', accuracy_score(y_test, y_pred))
    
    # Test in a sentence:
    sentence = st.text_input("Enter a sentence:")
    s = preprocessing(sentence)
    if st.button('Prediction'):
        pred = prediction(s, vectorizer, model)
        index = pred[0]
        if index == 0:
            st.write("=> Negative")
        elif index == 1:
            st.write("=> Neutral")
        else:
            st.write("=> Positive")
        
if __name__ == '__main__':
    main()


#0: negative, 1: neutral, 2: positive