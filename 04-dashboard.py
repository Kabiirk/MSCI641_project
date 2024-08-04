############################################################
#                                                          #
#                     PROJECT IMPORTS                      #
#                                                          #
############################################################
# For Dashboard creation
import streamlit as st

# Data Processing & Visualization
import pandas as pd
import requests
import plotly.express as px

# Web Scraping
from bs4 import BeautifulSoup

# for dataFrame.apply() optimization
import swifter

# NLP
import torch
import torchtext
from torchtext.data import get_tokenizer
torchtext.disable_torchtext_deprecation_warning()
from load_model_score import predict_sentiment, LSTM
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Other Utility ibraries
from collections import Counter

############################################################
#                                                          #
#                     HELPER FUNCTIONS                     #
#                                                          #
############################################################
# Download necessary NLTK data files
@st.cache_data
def load_nltk_resources():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

# Load LSTM Model & Vocabulary
@st.cache_resource
def load_model_and_vocab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer("basic_english")
    vocab = torch.load("models/02-final/vocab.pth")

    model = LSTM(
        vocab_size=9136,
        embedding_dim=300,
        hidden_dim=300,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout_rate=0.5,
        pad_index=1,
    )
    model.load_state_dict(torch.load("models/02-final/lstm_final.pt", map_location=torch.device('cpu')))
    model.to(device)
    return model, vocab, tokenizer, device

def print_something(text):
    url = 'https://www.imdb.com/title/'+text+'/'
    print("URL : ",url)
    return url

def is_valid_imdb_url(url):
    response = requests.get(url, headers = headers)
    print("Response: ",response)
    return response.status_code == 200

# For Responsible Scraping, identifies our scraper 
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
}

def scrape(id):
    start_url = 'https://www.imdb.com/title/'+id+'/reviews?ref_=tt_urv'
    link = 'https://www.imdb.com/title/'+id+'/reviews/_ajax'
    params = {
                 'ref_': 'undefined',
                 'paginationKey': ''
             }
    r_n = []
    source_code = requests.get(start_url)
    soup = BeautifulSoup(source_code.text, "lxml")
    movie_title = soup.find('title').get_text().split(' - ')[0]

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

        return r_n, movie_title

load_nltk_resources()
model, vocab, tokenizer, device = load_model_and_vocab()

############################################################
#                                                          #
#                UI LOGIC & DATA OPERATIONS                #
#                                                          #
############################################################

# movies_data = pd.read_csv("https://raw.githubusercontent.com/danielgrijalva/movie-stats/7c6a562377ab5c91bb80c405be50a0494ae8e582/movies.csv")
if 'movie_df' not in st.session_state:
    st.session_state.movie_df = pd.read_csv("reviews/01-cleaned_scored/VADER/cleaned_scored_reviews_tt0455944.csv")


st.session_state.movie_df.dropna()
# Convert the 'date' column to datetime format
st.session_state.movie_df['Date'] = pd.to_datetime(st.session_state.movie_df['Date'], format='%d %B %Y')
# Extract the year from the 'date' column
st.session_state.movie_df['year'] = st.session_state.movie_df['Date'].dt.year
st.session_state.movie_df['month'] = st.session_state.movie_df['Date'].dt.month

# Creating sidebar widget unique values from our movies dataset
if 'year_list' not in st.session_state:
    st.session_state.year_list = sorted(st.session_state.movie_df['year'].unique().tolist())

if 'movie_title' not in st.session_state:
    st.session_state.movie_title = "The Equalizer"

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

ID = st.text_input("## IMDb ID", "")
if st.button('Scrape Data'):
    with st.spinner(''):
        url = print_something(ID)
        progress_bar = st.progress(0)  # Initialize the progress bar
        status_text = st.empty()  # Placeholder for status text
        status_text.text('Scraping URL...')

        if is_valid_imdb_url(url):
            r_n, title = scrape(ID)

            progress_bar.progress(10)
            
            # convert scraped reviews to a datframe & drop rows with no reviews
            status_text.text('Creating Dataframe')
            review_df = pd.DataFrame(r_n)
            review_df = review_df[review_df['Review'].notna()]
            progress_bar.progress(20)
            
            status_text.text('Scoring')
            # Predict sentiment scores for all reviews
            review_df['Polarity'], review_df['Sentiment_Score'] = zip(*review_df['Review'].swifter.apply(lambda x: predict_sentiment(x, model, tokenizer, vocab, device)))
            # review_df['Predicted_Sentiment_Score'] = review_df['Review'].apply(lambda x: predict_sentiment(x, model, tokenizer, vocab, device)[1])
            progress_bar.progress(70)
            

            # Update placeholder Dataframe
            status_text.text('Updating Placeholders')
            st.session_state.movie_df = review_df
            print("Updated placeholder dataframe.")
            progress_bar.progress(80)
            
            # Convert the 'date' column to datetime format
            st.session_state.movie_df['Date'] = pd.to_datetime(st.session_state.movie_df['Date'], format='%d %B %Y')
            # Extract the year from the 'date' column
            st.session_state.movie_df['year'] = st.session_state.movie_df['Date'].dt.year
            st.session_state.year_list = sorted(st.session_state.movie_df['year'].unique().tolist())

            progress_bar.progress(90)
            status_text.text('Time conversions')

            # Extract Data to Update filter Slicers
            print("Updated sidebar Filters.")
            print("Final DF looks like this :")
            print(st.session_state.movie_df.head())
            status_text.text('Done !')
            progress_bar.progress(100)
            st.session_state.movie_title = title
            st.rerun()
        else:
            st.error('Failed to scrape data.')


############################################################
#                                                          #
#                GENERATING VISUALIZATIONS                 #
#                                                          #
############################################################

##############################
#                            #
#    HIGH-LEVEL OVERVIEW     #
#                            #
##############################
st.write(f"""## Movie Title : {st.session_state.movie_title}""")
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

st.write(f"##### **Total Reviews** : {len(st.session_state.movie_df)}")
st.write("##### **Sample review**")
st.write(st.session_state.movie_df['Review'][0])
st.write(f"##### **Author** : {st.session_state.movie_df['Author'][0]}")
st.write(f"##### ⭐ :  {st.session_state.movie_df['Stars(out_of_10)'][0]}")

##############################
#                            #
#         LINE CHART         #
#                            #
##############################
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


##############################
#                            #
#   DATA TABLE & PIE CHART   #
#                            #
##############################
# group the columns needed for visualizations
col1, col2 = st.columns([2,3])
with col1:
    st.write("""### Filtered Dataframe""")
    dataframe_genre_year = filtered_data
    dataframe_genre_year = dataframe_genre_year.reset_index(drop=True)
    st.dataframe(dataframe_genre_year, width = 400)

with col2:
    st.write(f"### Count of +ve & -ve reviews {new_year_rating[0]}-{new_year_rating[1]}")

    polarity_counts = filtered_data['Polarity'].value_counts().reset_index()
    polarity_counts.columns = ['Polarity', 'Count']
    # print(polarity_counts.head())
    fig = px.pie(polarity_counts, names='Polarity', values='Count',
                 labels={'Polarity': 'Polarity', 'Count': 'Count of Reviews'})
    st.plotly_chart(fig, theme=None)

##############################
#                            #
#          HEATMAP           #
#                            #
##############################
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

##############################
#                            #
#      WORD CLOUD/TREE       #
#                            #
##############################
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