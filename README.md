# News Sentiment & Category Analysis

News Sentiment & Category Analysis is a Data Science and NLP project built using Python and Streamlit.  
The application predicts news categories from headlines and performs sentiment analysis to classify news as positive, negative, or neutral.

## [Live Demo](https://news-sentiment-analysis-pred.streamlit.app/)


## Features

- News category prediction
- Sentiment analysis using VADER
- Confidence score display
- Interactive Streamlit interface
- NLP and Machine Learning based workflow

---

## Categories Supported

- Business
- Sports
- Politics
- Technology
- Entertainment

---

## Workflow

```text
News Headline
      ↓
Text Cleaning
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Category Prediction + Sentiment Analysis
```

---

## Model Performance

| Model | Accuracy |
|------|------|
| Naive Bayes | 72.92% |
| Logistic Regression | 80.44% |

---

## Technologies Used

- Python
- Scikit-learn
- NLTK
- Streamlit
- Pandas
- Matplotlib

---

## Dataset

Trained on more than 200,000 news articles across multiple categories.

---

## Project Structure

```text
NEWS_SENTIMENT_AND_ANALYSIS/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── runtime.txt
└── notebooks/
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/atharvmaske/news_sentiment_and_analysis.git
```

### Move into the project folder

```bash
cd news_sentiment_and_analysis
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/509c2ebb-4032-4bef-b7f8-955aaac2a46a" width="45%" />
  <img src="https://github.com/user-attachments/assets/f75cc23a-a176-42f2-9a30-976f6cd57b40" width="45%" />
</p>

---

## Built With

- Streamlit
- Scikit-learn
- Logistic Regression
- TF-IDF Vectorization
- VADER Sentiment Analysis
