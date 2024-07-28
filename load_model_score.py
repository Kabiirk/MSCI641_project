import torch
import torch.nn as nn
import torchtext
from torchtext.data import get_tokenizer
torchtext.disable_torchtext_deprecation_warning()

class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction

# unk_index = vocab["<unk>"]
# pad_index = vocab["<pad>"]
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

def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    if(predicted_class == 1):
        return [predicted_class, predicted_probability]
    else:
        return [predicted_class, -predicted_probability]
    
# def load_vocab(path):
#     return torch.load(path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = get_tokenizer("basic_english")
# vocab = load_vocab("models/baseline/vocab.pth")

# print(predict_sentiment("This film is terrible!", model, tokenizer, vocab, device))
# print(predict_sentiment("This film is great!", model, tokenizer, vocab, device))
# print(predict_sentiment("This film is not terrible, it's great!", model, tokenizer, vocab, device))

# import pandas as pd
# from collections import Counter
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.chunk import ne_chunk
# from collections import Counter

# # Sample dataframe
# data = {
#     'Title': ['Movie1', 'Movie2', 'Movie3', 'Movie4'],
#     'Author': ['Author1', 'Author2', 'Author3', 'Author4'],
#     'Date': pd.to_datetime(['2020-01-01', '2020-05-01', '2021-01-01', '2021-05-01']),
#     'Stars(out_of_10)': [7, 8, 9, 6],
#     'Review': ['Good movie with excellent acting', 'Excellent storyline and great visuals', 'Nice movie with good direction', 'Bad movie with poor acting'],
#     'Sentiment_Score': [0.7, 0.8, 0.9, 0.4],
#     'Polarity': [1, 1, 1, 0]
# }
# movie_df = pd.DataFrame(data)

# # Filter positive reviews
# positive_reviews = movie_df[movie_df['Polarity'] == 1]['Review']

# # Download necessary NLTK data files
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Function to preprocess and tokenize text
# def preprocess_text(text):
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text.lower())
#     filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
#     return filtered_words

# # Function to extract noun phrases
# def extract_noun_phrases(sentence):
#     words = word_tokenize(sentence)
#     pos_tags = nltk.pos_tag(words)
#     chunked = nltk.ne_chunk(pos_tags, binary=True)
    
#     noun_phrases = []
#     current_phrase = []
    
#     for subtree in chunked:
#         if type(subtree) == nltk.Tree:
#             current_phrase.append(" ".join([word for word, pos in subtree.leaves()]))
#         else:
#             if current_phrase:
#                 noun_phrases.append(" ".join(current_phrase))
#                 current_phrase = []
    
#     if current_phrase:
#         noun_phrases.append(" ".join(current_phrase))
    
#     return noun_phrases

# # Aggregate all positive reviews
# all_positive_reviews = ' '.join(positive_reviews)

# # Extract sentences
# sentences = sent_tokenize(all_positive_reviews)

# # Extract noun phrases from sentences
# sentence_features = []
# for sentence in sentences:
#     noun_phrases = extract_noun_phrases(sentence)
#     if noun_phrases:
#         sentence_features.extend(noun_phrases)

# # Count frequency of noun phrases
# noun_phrase_counts = Counter(sentence_features)

# # Get top 10 noun phrases
# top_noun_phrases = noun_phrase_counts.most_common(10)

# # Print top 10 noun phrases
# print("Top 10 features that people liked about the movie:")
# for phrase, count in top_noun_phrases:
#     print(f"{phrase}: {count}")