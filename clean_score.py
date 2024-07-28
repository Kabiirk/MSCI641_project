import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Download stopwords
nltk.download('vader_lexicon')
# nltk.download('stopwords')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # # Remove HTML tags
    # text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'/<[^>]*>/g', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces & trim trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # # Remove stopwords (optional)
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def get_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def get_class(score):
    if score>0:
        return 1
    else:
        return 0

def clean_and_score(id):
    # Load the dataset
    df = pd.read_csv('reviews_'+id+'.csv')
    # print(df.head)

    # Clean the reviews
    df['Review'] = df['Review'].apply(clean_text)
    # Get VADER sentiment scores
    df['Sentiment_Score'] = df['Review'].apply(get_vader_sentiment)
    # print(df.head)
    df['Polarity'] = df['Sentiment_Score'].apply(get_class)

    # print(df.head)

    # Write the cleaned data to a new CSV file
    df.to_csv('cleaned_scored/cleaned_scored_reviews_'+id+'.csv', index=False)

    print(f'Data cleaned and saved to cleaned_reviews{id}.csv')

id_list = [
            'tt0455944', # The Equalizer - ~750 reviews For testing
            'tt15398776', # Oppenheimer - 4K+ reviews supplementary data
            'tt0468569', # The Dark Knight - 9K+ reviews supplementary data
            'tt0111161' # The Shawshank Redemption - 11K+ reviews, for training
          ]

for id in id_list:
    clean_and_score(id)