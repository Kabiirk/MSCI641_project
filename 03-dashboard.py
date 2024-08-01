import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import requests, urllib
import plotly.express as px
import time
from bs4 import BeautifulSoup

# Loading the prediction model
import torch
import torchtext
from torchtext.data import get_tokenizer
torchtext.disable_torchtext_deprecation_warning()
from load_model_score import predict_sentiment, LSTM

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer("basic_english")
vocab = torch.load("models/baseline/vocab.pth")

model = LSTM(
    vocab_size = 8910,
    embedding_dim = 300,
    hidden_dim = 300,
    output_dim = 2,
    n_layers = 2,
    bidirectional = True,
    dropout_rate = 0.5,
    pad_index = 1,
)
model.load_state_dict(torch.load("models/baseline/lstm.pt", map_location=torch.device('cpu')))
# st.title('Counter Example')
# if 'count' not in st.session_state:
#     st.session_state.count = 0

# increment = st.button('Increment')
# if increment:
#     st.session_state.count += 1

# st.write('Count = ', st.session_state.count)

# movies_data = pd.read_csv("https://raw.githubusercontent.com/danielgrijalva/movie-stats/7c6a562377ab5c91bb80c405be50a0494ae8e582/movies.csv")
if 'movie_df' not in st.session_state:
    st.session_state.movie_df = pd.read_csv("test.csv")

# print("dF initialized Here")
# print(st.session_state.movie_df.head())

st.session_state.movie_df.dropna()
# Convert the 'date' column to datetime format
st.session_state.movie_df['Date'] = pd.to_datetime(st.session_state.movie_df['Date'], format='%d %B %Y')
# Extract the year from the 'date' column
st.session_state.movie_df['year'] = st.session_state.movie_df['Date'].dt.year
st.session_state.movie_df['month'] = st.session_state.movie_df['Date'].dt.month

# Creating sidebar widget unique values from our movies dataset
# score_rating = st.session_state.movie_df['Stars(out_of_10)'].unique().tolist()
# genre_list = st.session_state.movie_df['genre'].unique().tolist()
if 'year_list' not in st.session_state:
    st.session_state.year_list = sorted(st.session_state.movie_df['year'].unique().tolist())

sentiment_range = [-1.0, -0.5, 0.0, 0.5, 1.0]
polarity = ['Positive', 'Negative']

st.title('IMDb Viewer Opinion 🖖')
with st.sidebar:
    st.write("Select a range on the slider (it represents movie score) \
       to view the total number of movies in a genre that falls \
       within that range ")
    #create a slider to hold user scores
    new_score_rating = st.slider(label = "Choose a score:",
                                  min_value = -1.0,
                                  max_value = 1.0,
                                 value = (-1.0,1.0))
    #create a slider for movie years
    new_year_rating = st.slider(label = "Choose a Year Range:",
                                  min_value = min(st.session_state.year_list)-1,
                                  max_value = max(st.session_state.year_list),
                                  value=(min(st.session_state.year_list)-1, max(st.session_state.year_list)),
                                  step=1)

# Configure and filter the slider widget for interactivity
score_info = (st.session_state.movie_df['Sentiment_Score'].between(*new_score_rating))
date_info = (st.session_state.movie_df['year'] >= new_year_rating[0]) & (st.session_state.movie_df['year'] <= new_year_rating[1])
filtered_data = st.session_state.movie_df[score_info & date_info]

# print(type(score_info), type(movies_data), type(filtered_data))
# print(new_score_rating)
# print(new_year_rating)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
}

def print_something(text):
    url = 'https://www.imdb.com/title/'+text+'/'
    print("URL : ",url)
    return url

def is_valid_imdb_url(url):
    response = requests.get(url, headers = headers)
    print("Response: ",response)
    return response.status_code == 200

def scrape(id):
    start_url = 'https://www.imdb.com/title/'+id+'/reviews?ref_=tt_urv'
    link = 'https://www.imdb.com/title/'+id+'/reviews/_ajax'
    params = {
                 'ref_': 'undefined',
                 'paginationKey': ''
             }
    r_n = []
    with requests.Session() as s:
        s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
        res = s.get(start_url)

        while True:
            soup = BeautifulSoup(res.text,"lxml")
            for item in soup.select(".review-container"):
                # Scrape
                title = item.select_one("a.title").get_text(strip=True)
                author = item.select_one("span.display-name-link > a").get_text(strip=True)
                date = item.select_one("span.review-date").get_text(strip=True)
                stars = int(item.select_one("span.rating-other-user-rating > span").get_text()) if item.select_one("span.rating-other-user-rating > span") else None
                review = item.select_one("div.show-more__control").get_text(" ")
                permalink = item.select_one("div.text-muted > a")['href']

                # Process reviews
                review_cleaned = review.replace(',', ' ')

                # Poulate Review Object
                review = {
                    'Title':title,
                    'Author':author,
                    'Date':date,
                    'Stars(out_of_10)':stars,
                    'Review':review_cleaned
                }
                r_n.append(review)

            try:
                pagination_key = soup.select_one(".load-more-data[data-key]").get("data-key")
            except AttributeError:
                break
            params['paginationKey'] = pagination_key
            res = s.get(link,params=params)
            # time.sleep(1)

        return r_n

ID = st.text_input("## IMDb ID", "")
if st.button('Scrape Data'):
    with st.spinner('Scraping data...'):
        # time.sleep(2)  # Simulating a delay for demonstration purposes
        url = print_something(ID)

        if is_valid_imdb_url(url):
            r_n = scrape(ID)

            st.success('Scraping successful!')
            st.write(f"##### **Total Reviews** : {len(r_n)}")
            st.write("##### **Sample review**")
            st.write(r_n[0]['Review'])
            st.write(f"##### **Author** : {r_n[0]['Author']}")
            st.write(f"##### ⭐ :  {r_n[0]['Stars(out_of_10)']}")
            
            # convert scraped reviews to a datframe & drop rows with no reviews
            review_df = pd.DataFrame(r_n)
            review_df = review_df[review_df['Review'].notna()]
            print("Dataframe created.")
            st.success('Dataframe created.')
            
            # Predict sentiment scores for all reviews
            review_df['Polarity'], review_df['Sentiment_Score'] = zip(*review_df['Review'].apply(lambda x: predict_sentiment(x, model, tokenizer, vocab, device)))
            # review_df['Predicted_Sentiment_Score'] = review_df['Review'].apply(lambda x: predict_sentiment(x, model, tokenizer, vocab, device)[1])
            print("Scored new Reviews.")
            st.success('Scored new Reviews.')

            # Update placeholder Dataframe
            st.session_state.movie_df = review_df
            print("Updated placeholder dataframe.")
            st.success("Updated placeholder dataframe.")
            
            
            # Convert the 'date' column to datetime format
            st.session_state.movie_df['Date'] = pd.to_datetime(st.session_state.movie_df['Date'], format='%d %B %Y')
            # Extract the year from the 'date' column
            st.session_state.movie_df['year'] = st.session_state.movie_df['Date'].dt.year
            st.session_state.year_list = sorted(st.session_state.movie_df['year'].unique().tolist())

            # print("Inside:",st.session_state.year_list)
            print("Did time conversions.")
            st.success("Did time conversions.")

            # year_list = sorted(movies_data['year'].unique().tolist())

            # # Extract Data to Update filter Slicers
            # score_info = (movies_data['Sentiment_Score'].between(*new_score_rating))
            # date_info = (movies_data['year'] >= new_year_rating[0]) & (movies_data['year'] <= new_year_rating[1])
            # filtered_data = movies_data[score_info & date_info]
            print("Updated sidebar Filters.")
            st.success("Updated sidebar Filters.")

            print("Final DF looks like this :")
            print(st.session_state.movie_df.head())
            st.rerun()
            # print(predict_sentiment(r_n[0]['Review'], model, tokenizer, vocab, device))
        else:
            st.error('Failed to scrape data.')
# print_something(ID)

# visualization section

st.write("""## Should you watch this movie ?""")
# Calculate the average of the Sentiment_Score column
average_score = st.session_state.movie_df['Sentiment_Score'].mean()
# Define the threshold
threshold = 0.2

# Determine if the average score is above or below the threshold
result = ''
if(average_score >= 0.5):
    result = "### Definitely 🔥"
elif(average_score >= threshold and average_score < 0.5):
    result = "### Yes 😁"
elif(threshold - average_score < 0.1 and threshold - average_score > 0):
    result = "### Maybe 🤔"
else:
    result = "### No ☹️"

st.write(result)
st.write("Average Sentiment Score: ", average_score)

st.write("""## Average viewer sentiment across Years""")
if(len(st.session_state.year_list) == 1):
    # All reviews are from the same year i.e.
    # min(year_list) == max(year_list)
    rating_count_year = filtered_data.groupby('month')['Sentiment_Score'].mean()
    rating_count_year = rating_count_year.reset_index()
    figpx = px.line(rating_count_year, x = 'month', y = 'Sentiment_Score')
else:
    rating_count_year = filtered_data.groupby('year')['Sentiment_Score'].mean()
    rating_count_year = rating_count_year.reset_index()
    figpx = px.line(rating_count_year, x = 'year', y = 'Sentiment_Score')
st.plotly_chart(figpx)



# group the columns needed for visualizations
col1, col2 = st.columns([2,3])
with col1:
    st.write("""### Filtered Dataframe""")
    dataframe_genre_year = filtered_data
    dataframe_genre_year = dataframe_genre_year.reset_index(drop=True)
    st.dataframe(dataframe_genre_year, width = 400)

with col2:
    st.write(f"### Count of +ve & -ve reviews {new_year_rating[0]}-{new_year_rating[1]}")
    # rating_count_year = filtered_data.groupby('year')['Sentiment_Score'].mean()
    # rating_count_year = rating_count_year.reset_index()
    # figpx = px.line(rating_count_year, x = 'year', y = 'Sentiment_Score')
    # st.plotly_chart(figpx)

    polarity_counts = filtered_data['Polarity'].value_counts().reset_index()
    polarity_counts.columns = ['Polarity', 'Count']
    # print(polarity_counts.head())
    fig = px.pie(polarity_counts, names='Polarity', values='Count',
                 labels={'Polarity': 'Polarity', 'Count': 'Count of Reviews'})
    st.plotly_chart(fig, theme=None)

# Heatmap
# Group by year and month and calculate the average sentiment score
print("filtered_data.head()")
print(filtered_data.head())
avg_sentiment = filtered_data.groupby(['year', 'month'])['Sentiment_Score'].mean().reset_index()
print("avg_sentiment.head()")
print(avg_sentiment.head())

# Pivot for heatmap
st.write(f"## Average Sentiment Score by Month & Year")
pivot_table = avg_sentiment.pivot(index="year", columns="month", values="Sentiment_Score")
print("pivot_table.head()")
print(pivot_table.head())

heatmap_fig = px.imshow(
    pivot_table,
    labels=dict(x="month", y="year", color="Average Sentiment Score"),
    x=sorted(filtered_data['month'].unique().tolist()),
    y=pivot_table.index,
    color_continuous_scale='temps_r',
    aspect="auto"
)

heatmap_fig.layout.height = 600
heatmap_fig.layout.width = 600
st.plotly_chart(heatmap_fig, theme=None, use_container_width=True)

# Word clouds
filtered_data = filtered_data.dropna(subset=['Review'])
# People's likes and dislikes

positive_reviews = filtered_data.loc[filtered_data['Polarity'] == 1]
negative_reviews = filtered_data.loc[filtered_data['Polarity'] == 0]

# Function to extract nouns from a review
non_perm_words = ["movie", "series", "film"]
def extract_nouns(review):
    tokens = word_tokenize(review)
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('NN') and len(word)>3 and word.lower() not in non_perm_words]
    return nouns

# Extract nouns from reviews
positive_reviews['Nouns'] = positive_reviews['Review'].apply(extract_nouns)
negative_reviews['Nouns'] = negative_reviews['Review'].apply(extract_nouns)

# Flatten the list of nouns and count the frequency of each noun
all_positive_nouns = [noun for sublist in positive_reviews['Nouns'] for noun in sublist]
all_negative_nouns = [noun for sublist in negative_reviews['Nouns'] for noun in sublist]
positive_noun_counts = Counter(all_positive_nouns)
negative_noun_counts = Counter(all_negative_nouns)

# Create a DataFrame with the noun counts
positive_noun_df = pd.DataFrame(positive_noun_counts.items(), columns=['Noun', 'Frequency'])
positive_noun_df = positive_noun_df.sort_values(by=['Frequency'], ascending=False)
print(positive_noun_df.head(10))
negative_noun_df = pd.DataFrame(negative_noun_counts.items(), columns=['Noun', 'Frequency'])
negative_noun_df = negative_noun_df.sort_values(by=['Frequency'], ascending=False)
print(negative_noun_df.head(10))


st.write("""## What People Liked/Disliked""")
# Plot the word cloud using Plotly
st.write("""### The Good 😃""")
fig1 = px.treemap(positive_noun_df[:10], path=['Noun'], values='Frequency')
st.plotly_chart(fig1, theme=None, use_container_width=True)

st.write("""### The Bad 😡""")
fig2 = px.treemap(negative_noun_df[:10], path=['Noun'], values='Frequency')
st.plotly_chart(fig2, theme=None, use_container_width=True)