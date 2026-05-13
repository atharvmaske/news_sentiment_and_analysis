import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('vader_lexicon')
st.set_page_config(page_title="News")

with open("news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

analyzer = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

st.title("News")
st.write("News Category Classification & Sentiment Analysis")

headline = st.text_area("Enter News Headline")

if st.button("Analyze"):

    if headline.strip() == "":
        st.warning("Please enter a headline")

    else:
        cleaned = clean_text(headline)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]
        classes = model.classes_

        sentiment = get_sentiment(headline)

        st.subheader("Results")
        st.success(f"Predicted Category: {prediction}")
        st.info(f"Sentiment: {sentiment}")

        st.subheader("Category Confidence")

        top_indices = probabilities.argsort()[-5:][::-1]

        for idx in top_indices:
            category = classes[idx]
            confidence = probabilities[idx] * 100

            st.write(f"{category} : {confidence:.2f}%")
            st.progress(float(probabilities[idx]))