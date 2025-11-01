# app.py - Streamlit Music Recommender with preprocessing
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------- SPOTIFY CONFIG -----------------
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    try:
        # Try with both song + artist first
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track", limit=1)
        
        # If not found, retry with just the song name
        if not results["tracks"]["items"]:
            results = sp.search(q=f"track:{song_name}", type="track", limit=1)

        if results and results["tracks"]["items"]:
            return results["tracks"]["items"][0]["album"]["images"][0]["url"]
        else:
            return "https://i.postimg.cc/0QNxYz4V/social.png"
    except Exception:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

# ----------------- MUSIC RECOMMENDER -----------------
class MusicRecommender:
    def __init__(self, csv_path="small_spotify_sample.csv"):
        self.df = None
        self.similarity = None
        self.stemmer = PorterStemmer()
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Load preprocessed data if exists
        if os.path.exists('df.pkl') and os.path.exists('similarity.pkl'):
            self.load_model()
        else:
            self.preprocess(csv_path)

    def preprocess(self, csv_path):
        if not os.path.exists(csv_path):
            st.error(f"CSV file '{csv_path}' not found!")
            return
        self.df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
        # Drop extra columns if exist
        if 'link' in self.df.columns:
            self.df = self.df.drop('link', axis=1)
        # Optional: sample 5000 songs
        if len(self.df) > 5000:
            self.df = self.df.sample(5000, random_state=42).reset_index(drop=True)
        # Text preprocessing
        self.df['text'] = self.df['text'].str.lower().replace(r'\n', ' ', regex=True)
        self.df['text'] = self.df['text'].replace(r'[^\w\s]', ' ', regex=True)
        self.df['text'] = self.df['text'].fillna("")

        # Tokenization + stemming
        self.df['text'] = self.df['text'].apply(self.tokenize_and_stem)

        # Create similarity matrix
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['text'])
        self.similarity = cosine_similarity(tfidf_matrix)

        # Save for later
        with open('df.pkl', 'wb') as f:
            pickle.dump(self.df, f)
        with open('similarity.pkl', 'wb') as f:
            pickle.dump(self.similarity, f)

    def tokenize_and_stem(self, txt):
        tokens = nltk.word_tokenize(str(txt))
        stems = [self.stemmer.stem(t) for t in tokens if t.isalpha()]
        return " ".join(stems)

    def load_model(self):
        with open('small_similarity.pkl', 'rb') as f:
            self.similarity = pickle.load(f)
        self.df = pd.read_csv('small_spotify_sample.csv', on_bad_lines='skip', engine='python')

    def recommend(self, song_name, n=5):
        if song_name not in self.df['song'].values:
            return [], []
        idx = self.df[self.df['song'] == song_name].index[0]
        distances = sorted(list(enumerate(self.similarity[idx])), reverse=True, key=lambda x: x[1])
        recommended_names, recommended_posters = [], []
        for i in distances[1:n+1]:
            rec_song = self.df.iloc[i[0]]['song']
            rec_artist = self.df.iloc[i[0]]['artist']
            recommended_names.append(rec_song)
            recommended_posters.append(get_song_album_cover_url(rec_song, rec_artist))
        return recommended_names, recommended_posters

# ----------------- STREAMLIT UI -----------------
st.header("🎵 Music Recommender System")

recommender = MusicRecommender()

song_list = recommender.df['song'].values if recommender.df is not None else []
selected_song = st.selectbox("Type or select a song from the dropdown", song_list)

if st.button("Show Recommendations"):
    names, posters = recommender.recommend(selected_song)
    if names:
        cols = st.columns(5)
        for i in range(len(names)):
            with cols[i]:
                st.text(names[i])
                st.image(posters[i])
    else:
        st.warning("No recommendations found!")
