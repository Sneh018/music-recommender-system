# ✅ app.py — Final Streamlit Cloud Fix (punkt_tab issue solved)
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(page_title="🎵 Music Recommender System", layout="wide")
st.markdown("""
<style>
code, pre { white-space: pre-wrap; word-break: break-word; }
</style>
""", unsafe_allow_html=True)

# ----------------- NLTK SETUP (safe for Streamlit Cloud) -----------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ----------------- SPOTIFY CONFIG -----------------
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_song_album_cover_url(song_name, artist_name):
    """Fetch album cover from Spotify API; fallback if not found."""
    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track", limit=1)
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

        if os.path.exists('df.pkl') and os.path.exists('similarity.pkl'):
            self.load_model()
        else:
            self.preprocess(csv_path)

    def preprocess(self, csv_path):
        """Load, clean, and preprocess dataset."""
        if not os.path.exists(csv_path):
            st.error(f"CSV file '{csv_path}' not found!")
            return

        st.info("📂 Loading and processing dataset...")
        self.df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')

        if 'link' in self.df.columns:
            self.df = self.df.drop('link', axis=1)

        if len(self.df) > 5000:
            self.df = self.df.sample(5000, random_state=42).reset_index(drop=True)

        self.df['text'] = self.df['text'].fillna("").str.lower()
        self.df['text'] = self.df['text'].replace(r'\n', ' ', regex=True)
        self.df['text'] = self.df['text'].replace(r'[^\w\s]', ' ', regex=True)

        self.df['text'] = self.df['text'].apply(self.tokenize_and_stem)

        tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['text'])
        self.similarity = cosine_similarity(tfidf_matrix)

        with open('df.pkl', 'wb') as f:
            pickle.dump(self.df, f)
        with open('similarity.pkl', 'wb') as f:
            pickle.dump(self.similarity, f)

        st.success("✅ Dataset processed successfully!")

    def tokenize_and_stem(self, txt):
        """Tokenize text safely — fixes punkt_tab LookupError."""
        try:
            tokens = word_tokenize(str(txt))
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            tokens = word_tokenize(str(txt))
        except Exception:
            tokens = str(txt).split()

        stems = [self.stemmer.stem(t) for t in tokens if t.isalpha()]
        return " ".join(stems)

    def load_model(self):
        with open('similarity.pkl', 'rb') as f:
            self.similarity = pickle.load(f)
        with open('df.pkl', 'rb') as f:
            self.df = pickle.load(f)

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
st.header("🎶 Music Recommender System")
st.markdown("##### Get personalized song suggestions based on lyrics and Spotify visuals 🎧")

recommender = MusicRecommender()
song_list = recommender.df['song'].values if recommender.df is not None else []

selected_song = st.selectbox("🎵 Choose or type a song", song_list)

if st.button("Show Recommendations"):
    names, posters = recommender.recommend(selected_song)
    if names:
        cols = st.columns(len(names))
        for i, col in enumerate(cols):
            with col:
                st.text(names[i])
                st.image(posters[i])
    else:
        st.warning("⚠️ No recommendations found for this song.")
