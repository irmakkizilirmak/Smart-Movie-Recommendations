import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 1. DATA AND AI
# ==========================================

# Cache the data loading process so it doesn't reload on every user interaction (improves speed)
@st.cache_data  
def prepare_data():
    # Reading local CSV files directly from the repository instead of Google Drive links
    movies = pd.read_csv('movies.csv')
    credits = pd.read_csv('credits.csv')
    credits.columns = ['movie_id', 'title', 'cast', 'crew']
    
    # Merge the dataframes and drop duplicate titles to avoid conflicts
    df = movies.merge(credits, on='title').drop_duplicates(subset=['title']).reset_index(drop=True)
    
    # Regex function to clean and extract names from JSON-like fields (genres, keywords)
    def clean_text(text):
        if pd.isna(text): 
            return ""
        found = re.findall(r'"name":\s*"([^"]+)"', str(text))
        return " ".join([i.lower().replace(" ", "") for i in found])
        
    df['clean_genres'] = df['genres'].apply(clean_text)
    df['clean_keywords'] = df['keywords'].apply(clean_text)
    df['overview'] = df['overview'].fillna('')
    
    # Combine all metadata into a single string column (metadata soup)
    df['info_soup'] = df['clean_genres'] + " " + df['clean_keywords'] + " " + df['overview'].str.lower()
    return df

# Initialize data preparation
df = prepare_data()

# Calculate TF-IDF matrix and pairwise cosine similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['info_soup'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series to quickly lookup a movie's index by its title
movie_indices = pd.Series(df.index, index=df['title'])

# Defined 6 dimensions of cinema with their respective target keywords
cinema_dimensions = {
    "Dramatik Yapı ve Duygu": ["drama", "emotion", "sad", "love", "family", "relationship", "tragedy", "crying", "feeling", "heart", "betrayal"],
    "Görsel ve İşitsel Estetik": ["cinematography", "visual", "soundtrack", "music", "color", "atmosphere", "beautiful", "effects", "aesthetic"],
    "Teknik Başarı ve Zanaat": ["director", "editing", "camera", "sound", "production", "oscar", "award", "masterpiece", "technique"],
    "Zihinsel Egzersiz ve Senaryo": ["twist", "mystery", "philosophical", "mind", "puzzle", "complex", "psychological", "metaphor", "intelligent"],
    "Escapizm (Gerçeklikten Kaçış)": ["sci-fi", "fantasy", "alien", "magic", "adventure", "action", "space", "future", "monster", "superhero"],
    "Sosyokültürel Keşif": ["history", "culture", "society", "biography", "war", "documentary", "politics", "world", "human", "period"]
}

# ==========================================
# 2. STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Gurme Sinema Öneri Sistemi", page_icon="🎬", layout="centered")
st.title("🎬 Yapay Zeka Tabanlı Gurme Sinema Asistanı")
st.write("İzlediğiniz bir filmi seçin ve o filmde sizi en çok etkileyen boyutu işaretleyin. Yapay zeka size nokta atışı gurme öneriler sunsun!")

# Setup UI input selectors
movie_list = sorted(df['title'].tolist())
selected_movie = st.selectbox("İzlediğiniz ve beğendiğiniz film:", movie_list)
dimensions = list(cinema_dimensions.keys())
selected_dimension = st.radio("Bu filmde en çok hangi yön ilginizi çekti?", dimensions)

if st.button("Gurme Önerileri Getir 🎯"):
    if selected_movie in movie_indices:
        idx = movie_indices[selected_movie]
        dimension_words = cinema_dimensions[selected_dimension]
        
        # Extract AI similarity scores for the chosen movie
        scores = list(enumerate(similarity_matrix[idx]))
        
        # Sort all movies based on similarity scores in descending order
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Extract the top 40 candidate movies (excluding the selected movie itself, starting at index 1)
        top_40_candidates = scores[1:41]
        top_40_indices = [i for i, score in top_40_candidates]
        
        # Store AI similarity scores in a dictionary mapping for final sorting sorting purposes
        tfidf_scores_map = {i: score for i, score in top_40_candidates}
        candidate_movies = df.iloc[top_40_indices].copy()
        
        # Exact-word matching function using regex to count dimension keywords safely 
        # (e.g., skips matching 'love' inside words like 'glover')
        def calculate_dimension_score(info_soup):
            score = 0
            for word in dimension_words:
                score += len(re.findall(r'\b' + re.escape(word.lower()) + r'\b', info_soup))
            return score
            
        candidate_movies['dimension_score'] = candidate_movies['info_soup'].apply(calculate_dimension_score)
        candidate_movies['tfidf_score'] = candidate_movies.index.map(tfidf_scores_map)
        
        # Primary sort by chosen user dimension score, secondary sort by underlying AI similarity score
        results = candidate_movies.sort_values(by=['dimension_score', 'tfidf_score'], ascending=[False, False]).head(5)
        
        st.success(f"🍿 '{selected_movie}' filmini sevenler ve odağı [{selected_dimension}] olanlar için yapay zeka sonuçları:")
        
        # Display the final recommended results on screen
        for i, (_, row) in enumerate(results.iterrows(), 1):
            # Formats raw comma-less space string tags into pretty capitalized tags
            genres_list = ", ".join(row['clean_genres'].split()).title() if row['clean_genres'] else "Not Specified"
            
            # Convert decimal similarity score to an easy-to-read percentage 
            similarity_percentage = round(row['tfidf_score'] * 100, 1)
            
            st.markdown(f"**{i}. {row['title']}**")
            st.caption(f"🎥 Türler: {genres_list} | 🎯 Odak Skoru: {row['dimension_score']} | 🤖 Yapay Zeka Benzerliği: %{similarity_percentage}")
            st.divider()
