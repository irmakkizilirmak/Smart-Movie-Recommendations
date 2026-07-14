import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests

# ==========================================
# 0. API AND POSTER
# ==========================================
def get_poster_url(movie_id):
    """
    by using TMDB API 
    """
    try:
       
        api_key = st.secrets["TMDB_API_KEY"] 
        url = f"https://themoviedb.org{movie_id}?api_key={api_key}"
        
        response = requests.get(url, timeout=2).json()
        poster_path = response.get('poster_path')
        if poster_path:
          
            return f"https://tmdb.org{poster_path}"
    except Exception:
        
        pass
    return "https://placeholder.com"


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
    "Story & Intense Emotion": ["drama", "emotion", "sad", "love", "family", "relationship", "tragedy", "crying", "feeling", "heart", "betrayal"],
    "Cinematography & Atmosphere": ["cinematography", "visual", "soundtrack", "music", "color", "atmosphere", "beautiful", "effects", "aesthetic"],
    "Direction & Craft": ["director", "editing", "camera", "sound", "production", "oscar", "award", "masterpiece", "technique"],
    "Mind Games & Plot Twists": ["twist", "mystery", "philosophical", "mind", "puzzle", "complex", "psychological", "metaphor", "intelligent"],
    "Fantasy Worlds & Escapism": ["sci-fi", "fantasy", "alien", "magic", "adventure", "action", "space", "future", "monster", "superhero"],
    "Period & Real History": ["history", "culture", "society", "biography", "war", "documentary", "politics", "world", "human", "period"]
}

# ==========================================
# 2. STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Gourmet Movie Recommender System", page_icon="🎬", layout="centered")
st.title("AI-Powered Cinema Assistant")
st.write("Select a movie you have watched and choose the aspect that impressed you the most. Let AI provide you with pinpoint gourmet recommendations!")

# Setup UI input selectors
movie_list = sorted(df['title'].tolist())
selected_movie = st.selectbox("Select a movie you watched and liked:", movie_list)
dimensions = list(cinema_dimensions.keys())
selected_dimension = st.radio("Which aspect of this movie captivated you the most?", dimensions)

if st.button("Discover Gourmet Matches"):
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
        
        # Store AI similarity scores in a dictionary mapping for final sorting purposes
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
        
        st.success(f"🍿 '{selected_movie}' AI results for those who love [{selected_dimension}] and focus on:")
        
        # Display the final recommended results on screen
        for i, (_, row) in enumerate(results.iterrows(), 1):
            # Formats raw comma-less space string tags into pretty capitalized tags
            genres_list = ", ".join(row['clean_genres'].split()).title() if row['clean_genres'] else "Not Specified"
            
            # Convert decimal similarity score to an easy-to-read percentage 
            similarity_percentage = round(row['tfidf_score'] * 100, 1)
            
            # Ekran düzenini 2 sütuna ayırıyoruz: Afiş alanı (%30) ve Detay alanı (%70)
            col1, col2 = st.columns([1, 2.3])
            
            with col1:
                # TMDB API'den film afişini çekip ekrana basıyoruz
                poster_url = get_poster_url(row['movie_id'])
                st.image(poster_url, use_container_width=True)
                
            with col2:
                st.markdown(f"### {i}. {row['title']}")
                st.caption(f"**Genres:** {genres_list}")
                
                # Eşleşme detaylarını gurme indikatörlerle gösteriyoruz
                st.write(f"🎯 Dimension Score: `{row['dimension_score']}`")
                
                # AI benzerlik uyumunu şık bir ilerleme çubuğu (progress bar) ile gösterelim
                st.write("🤖 AI Similarity Match:")
                st.progress(float(row['tfidf_score']) if row['tfidf_score'] <= 1.0 else 1.0)
                st.caption(f"Match Rate: %{similarity_percentage}")
                
                # Film özetini açılır kutu (expander) içine gizleyerek temiz tasarım sağlıyoruz
                if 'overview' in row and row['overview']:
                    with st.expander("🎞️ Read Overview"):
                        st.write(row['overview'])
                        
            st.divider()
