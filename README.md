# MSCI641_project
SENTIMENT ANALYSIS OF IMBD Movie REVIEWS USING LTSM (Long Short-Term Memory) Neural Networks  

1. **MEMBER 1**
   - **Name:** Kabiir Krishna
   - **Email:** k7krishn@uwaterloo.ca
   - **WatIAM:** k7krishn
   - **Student Number:** 21106092

2. **MEMBER 2**
   - **Name:** Akshat Baheti
   - **Email:** abahieti@uwaterloo.ca
   - **WatIAM:** abahieti
   - **Student Number:** 21100660

## About the Project
In the digital age, online reviews significantly
influence decisions about movies and TV
shows. This project explores the use of **Long
Short-Term Memory (LSTM)** neural networks
for sentiment analysis of IMDB reviews. Using
**distilBERT** and **VADER**, we generate continuous 
sentiment scores ranging from `-1` to `+1` for
our training dataset. These scores train the
LSTM model to handle the sequential nature of
textual data, accurately identifying consensus
and overall sentiment across reviews.

Our approach helps the entertainment industry & users
understand **audience preferences**, guiding
**marketing strategies**, **recommendation systems**,
and content creation. Consumers benefit from
the wisdom of the crowd, which helps them
make better choices.

The technology can also
be extended to other areas such as **product
reviews** and **social media monitoring**.
Experiments show that our model effectively
captures and analyzes sentiment from large-scale data, demonstrating the potential of
sentiment analysis to improve decision-making
and tailor content to audience expectations.
_______

## Getting Started

### Project Structure
```
Project Root/
│   .gitignore
│   00-scrape.py
│   01.1-clean_score.py
│   01.2-Training_DistilBERT.ipynb
│   02-msci641_project.ipynb
│   03-sentiment_finetuning_w_distilbert.ipynb
│   04-dashboard.py
│   load_model_score.py
│   requirements.txt
│   README.md
│
├───media
│       # Contains media assets for README.md
│
├───models
│   ├───00-baseline
│   │       lstm.pt
│   │       vocab.pth
│   │
│   └───02-final
│           lstm_final.pt
│           vocab.pth
│
├───project_reports
│       00-Project Proposal.pdf
│       01-Project Milestone.pdf
│
└───reviews
    ├───00-scraped
    │       reviews_tt0111161.csv
    │       reviews_tt0455944.csv
    │       reviews_tt0468569.csv
    │       reviews_tt15398776.csv
    │
    └───01-cleaned_scored
        ├───VADER
        │       cleaned_scored_reviews_tt0111161.csv
        │       cleaned_scored_reviews_tt0455944.csv
        │       cleaned_scored_reviews_tt0468569.csv
        │       cleaned_scored_reviews_tt15398776.csv
        │
        └───VADER_DISTILBERT_FINAL
                vader_dbert_scored.csv
```

### Important Files/Folders

#### Preparing training Data
1. `00-scrape.py`: Contains logic for Scraping reviews (This script was used to generate initial scraped data which was later cleaned and socred with 01-clean_score.py :). To run it separately, type:
    ```
    python3 00-scrape.py
    ```
2. `01.1-clean_score.py`: Cleans scraped reviews & scores them with VADER to create. (Works on the data scraped by previous script, cleans it, scores it using VADER and outputs cleaned & scored `.csv` files in `01-cleaned_scored/VADER/`). Can be run separately using:
    ```
    python3 01.1-scrape.py
    ```
#### Training the DistilBERT & LSTM Model
1. `01.2-Training_DistilBERT.ipynb`: Notebooko for Training DistilBERT model, a strong classifier.
2. `02-msci641_project.ipynb :` Training Script for the LSTM model. Exports the model for future usage too.
3. `load_model_score.py :` Contains LSTM definintion (Helps with loading) & Function to score using a pre-loaded model.
4. `04-dashboard.py :` Contains main Logic for dashboard webpage displayed.

#### Augmenting data wth scores from VADER and DistilBERT:
1. `03-sentiment_finetuning_w_distilbert.ipynb`: Contains logic for loading DistilBERT model trained with `01.2-Training_DistilBERT.ipynb` & using it to augment our VADER-scored reviews.

#### Data Folders
1. `reviews/00-scraped/`: Contains the initial reviews scraped by Web Crawler.
2. `reviews/01-cleaned_scored/VADER/`: contains the cleaned reviews which were only scored by VADER.
3. `reviews/01-cleaned_scored/VADER_DISTILBERT_FINAL/`: Contains the Reivews which were scored by DistilBERT and the final finetuned (VADER + DistilBERT) scores for the review. This was the final training data used to train the model.
4. `models/`: Contans the Initial & the Final LSTM models used in this project. (The DistilBERT model couldn't be included in this repo since it was too large to be pushed here)
> **NOTE:** Though not required to run the project, the DistilBERT model can be regenerated at user's end by executing the `01.2-Training_DistilBERT.ipynb`.

### How To Run the Project locally
1. Clone the repository
    ```
    git clone https://github.com/Kabiirk/MSCI641_project.git
    ```

2. Navigate to the project root
   ```
   cd MSCI641_project
   ```

3. Install dependdencies
   ```
   pip install -r requirements.txt
   ```

4. Run the Dashboard
   ```
   streamlit run 04-dashboard.py
   ```
   This would automatically open a new Browser window/tab with the dashboard deployed on `localhost`.

### Using the Dashboard
Upon running the project initially, the Dashboard loads up the pre-scraped reviews of the movie ["The Equalizer"](https://www.imdb.com/title/tt0455944/) so that the sample visualizations are already visible. The users can start real-time scraping and analnysis of any new movie by following these steps:

1. Type out the movie/TV-Series ID as per [IMDb](https://www.imdb.com/) in the Text boc (under "IMDb ID") at the top and press the `Scrape Data` button.
2. A live progess Bar will apprear indicating the status of operations (`Scraping`,`Scoring` etc.)
3. Once the Reviews are loaded & analysis is done, the Dashboard will update the existing visualization as per the new reviews which have been scored by our model.
_______

### Demo Video
![Project Recording](./media/msci641_project_recording.gif)

> **Note:** The Demo video has been trimmed (& sped up) for brevity. This project scrapes the latest reviews every time the user inputs a Movie/TV-Show ID for analysis by scraping the reviews afresh. Upon creating t dataframe of the reviews, the script uses Pandas' `apply()` function ([ref.](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html)) on the `Reviews` column of the new Dataframe which can take some time depending on the compute resources available on the local machine running the dashboard. Ideally, for quick results, it is suggested to scrape for movies with fewer reviews.