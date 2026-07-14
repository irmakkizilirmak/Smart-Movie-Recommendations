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
    TMDB API v3 formatına uygun olarak güncel afiş linkini çeker.
    """
    try:
        
        clean_id = int(float(movie_id))
        
        api_key = st.secrets["TMDB_API_KEY"] 
        url = f"https://themoviedb.org{clean_id}?api_key={api_key}"
        
        response = requests.get(url, timeout=2).json()
        poster_path = response.get('poster_path')
        if poster_path:
            return f"https://tmdb.org{poster_path}"
    except Exception:
        pass
 
    return "https://placeholder.com🎬+Afi%C5%9F+Yüklenemedi"



# ==========================================
# 1. DATA AND AI
# ==========================================

# CODE 1: ESKİ HAZIRLIK KISMINI SİLİP BUNU YAZIN (KODUN ÜST KISMI)
def fetch_recommendation_pool(movie_id):
    """
    Seçilen filmin benzerlerini API'den canlı çekerek 
    AI modelimiz için dinamik bir 'Veri Seti' oluşturur.
    """
    api_key = st.secrets["TMDB_API_KEY"]
    pool = []
    seen_ids = set()
    
    # TMDB API'den 4 farklı kanaldan filmleri topluyoruz
    urls = [
        f"https://themoviedb.org{movie_id}/similar?api_key={api_key}&language=en-US&page=1",
        f"https://themoviedb.org{movie_id}/recommendations?api_key={api_key}&language=en-US&page=1",
        f"https://themoviedb.orgtop_rated?api_key={api_key}&language=en-US&page=1",
        f"https://themoviedb.orgpopular?api_key={api_key}&language=en-US&page=1"
    ]
    
    for url in urls:
        try:
            results = requests.get(url, timeout=3).json().get('results', [])
            for m in results:
                if m['id'] not in seen_ids and m.get('overview'):
                    seen_ids.add(m['id'])
                    pool.append({
                        'movie_id': m['id'],
                        'title': m['title'],
                        'overview': m['overview'],
                        'vote_average': m.get('vote_average', 0.0),
                        'poster_path': m.get('poster_path', ''),
                        'clean_genres': "" # Canlıda türleri karmaşıklaştırmamak için boş geçiyoruz
                    })
        except:
            continue
            
    return pd.DataFrame(pool)


cinema_dimensions = {
    "Story & Intense Emotion": ["drama", "emotion", "sad", "love", "family", "relationship", "tragedy", "crying", "feeling", "heart", "betrayal"],
    "Cinematography & Atmosphere": ["cinematography", "visual", "soundtrack", "music", "color", "atmosphere", "beautiful", "effects", "aesthetic"],
    "Direction & Craft": ["director", "editing", "camera", "sound", "production", "oscar", "award", "masterpiece", "technique"],
    "Mind Games & Plot Twists": ["twist", "mystery", "philosophical", "mind", "puzzle", "complex", "psychological", "metaphor", "intelligent"],
    "Fantasy Worlds & Escapism": ["sci-fi", "fantasy", "alien", "magic", "adventure", "action", "space", "future", "monster", "superhero"],
    "Period & Real History": ["history", "culture", "society", "biography", "war", "documentary", "politics", "world", "human", "period"]
}
# ==========================================
# 2. STREAMLIT INTERFACE & LIVE AI ENGINE
# ==========================================
st.title("AI-Powered Cinema Assistant (Live Edition)")
st.write("Search across millions of movies and let our gourmet AI model provide you with pinpoint recommendations!")

# Step 1: Live Movie Search Input
search_query = st.text_input("Select a movie you watched and liked (e.g., The Love Witch, Saw, Batman Begins):", "")

if search_query:
    api_key = st.secrets["TMDB_API_KEY"]
    search_url = f"https://themoviedb.org{api_key}&query={search_query}&language=en-US"
    
    try:
        search_results = requests.get(search_url, timeout=3).json().get('results', [])
    except:
        search_results = []
    
    if search_results:
        # Fill the selectbox with live matching movies from the API
        movie_options = {f"{m['title']} ({m.get('release_date', '')[:4]})": m for m in search_results}
        selected_display = st.selectbox("Which movie do you mean exactly?", list(movie_options.keys()))
        
        target_movie = movie_options[selected_display]
        
        # Setup UI input selectors for dimensions
        dimensions = list(cinema_dimensions.keys())
        selected_dimension = st.radio("Which aspect of this movie captivated you the most?", dimensions)
        
        # Step 2: Live Gourmet Button Trigger
        if st.button("Discover Gourmet Matches"):
            with st.spinner("Gourmet AI model is training on the dynamic dataset..."):
                
                # Fetch live dynamic pool from the API based on the selected movie ID
                df_pool = fetch_recommendation_pool(target_movie['id'])
                
                if not df_pool.empty:
                    # Insert the original target movie right at index 0 for the AI reference
                    orig_row = pd.DataFrame([{
                        'movie_id': target_movie['id'],
                        'title': target_movie['title'],
                        'overview': target_movie['overview'],
                        'vote_average': target_movie.get('vote_average', 0.0),
                        'poster_path': target_movie.get('poster_path', ''),
                        'clean_genres': ""
                    }])
                    df_pool = pd.concat([orig_row, df_pool]).drop_duplicates(subset=['movie_id']).reset_index(drop=True)
                    
                    # --- REAL-TIME AI PIPELINE (TF-IDF & COSINE SIMILARITY) ---
                    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = tfidf_vectorizer.fit_transform(df_pool['overview'].str.lower())
                    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
                    
                    # Extract AI similarity scores (Orijinal film is always at index 0)
                    scores = list(enumerate(similarity_matrix))
                    scores = sorted(scores, key=lambda x: x, reverse=True)
                    top_candidates = scores[1:]
                    
                    valid_indices = []
                    tfidf_scores_map = {}
                    seen_overviews = set()
                    
                    # Store original parameters for advanced duplicate & sequel filtering
                    selected_title_lower = target_movie['title'].lower()
                    selected_title_words = set([w for w in selected_title_lower.split() if len(w) > 2])
                    selected_overview_clean = target_movie['overview'].strip().lower()
                    
                    if selected_overview_clean:
                        seen_overviews.add(selected_overview_clean)
                        
                    for idx, score in top_candidates:
                        candidate_row = df_pool.iloc[idx]
                        candidate_title = candidate_row['title'].lower()
                        candidate_overview = candidate_row['overview'].strip().lower()
                        
                        # --- 1. STRICT DUPLICATE OVERVIEW FILTER ---
                        if candidate_overview in seen_overviews or any(candidate_overview[:30] == seen[:30] for seen in seen_overviews):
                            continue
                            
                        # --- 2. GLOBAL SEQUEL & SAME UNIVERSE FILTER ---
                        candidate_title_words = set([w for w in candidate_title.split() if len(w) > 2])
                        is_same_name = bool(selected_title_words.intersection(candidate_title_words))
                        
                        if is_same_name:
                            continue # Skip direct sequels or franchise matches
                            
                        valid_indices.append(idx)
                        tfidf_scores_map[idx] = score
                        if candidate_overview:
                            seen_overviews.add(candidate_overview)
                            
                    candidate_movies = df_pool.iloc[valid_indices].copy().reset_index(drop=True)
                    
                    # Exact-word matching function using regex to count dimension keywords safely
                    dimension_words = cinema_dimensions[selected_dimension]
                    def calculate_dimension_score(overview):
                        score = 0
                        for word in dimension_words:
                            score += len(re.findall(r'\b' + re.escape(word.lower()) + r'\b', str(overview).lower()))
                        return score
                        
                    candidate_movies['dimension_score'] = candidate_movies['overview'].apply(calculate_dimension_score)
                    candidate_movies['tfidf_score'] = candidate_movies['title'].map({df_pool.loc[k, 'title']: s for k, s in tfidf_scores_map.items()})
                    
                    # IMDb 6.0 Penalty Safeguard
                    def apply_imdb_penalty(row):
                        if row['vote_average'] < 6.0:
                            return float(row['dimension_score']) * 0.5
                        return float(row['dimension_score'])
                    candidate_movies['final_dimension_score'] = candidate_movies.apply(apply_imdb_penalty, axis=1)
                    
                    # Quality Score and Final AI-Ordered Ranking
                    candidate_movies['quality_score'] = (candidate_movies['tfidf_score'] * 0.4) + ((candidate_movies['vote_average'] / 10.0) * 0.6)
                    results = candidate_movies.sort_values(by=['tfidf_score', 'quality_score'], ascending=[False, False]).head(5)
                    
                    # ==========================================
                    # 3. DISPLAY RESULTS (IMPECCABLE DESIGN)
                    # ==========================================
                    with st.container():
                        st.success(f"🍿 '{target_movie['title']}' AI results for those who love [{selected_dimension}]:")
                        
                        for i, (_, row) in enumerate(results.iterrows(), 1):
                            tf_val = row['tfidf_score']
                            similarity_percentage = round(tf_val * 100, 1) if pd.notna(tf_val) else 0.0
                            
                            col1, col2 = st.columns([1, 2.3])
                            
                            with col1:
                                if row['poster_path']:
                                    st.image(f"https://tmdb.org{row['poster_path']}", use_container_width=True)
                                else:
                                    st.image("https://placeholder.com🎬+No+Poster", use_container_width=True)
                                
                            with col2:
                                st.markdown(f"### {i}. {row['title']}")
                                st.markdown(f"⭐ **IMDb:** `{round(row['vote_average'], 1)}/10`")
                                
                                if row['vote_average'] < 6.0:
                                    st.write(f"🎯 Match Score: `{int(row['dimension_score'])}` (Penalized: `{row['final_dimension_score']}`)")
                                else:
                                    st.write(f"🎯 Match Score: `{int(row['dimension_score'])}`")
                                    
                                st.write("🤖 AI Similarity Match:")
                                progress_value = float(tf_val) if pd.notna(tf_val) and tf_val <= 1.0 else 0.0
                                st.progress(progress_value)
                                st.caption(f"Match Rate: %{similarity_percentage}")
                                
                                if row['overview']:
                                    with st.expander("Read Overview"):
                                        st.write(row['overview'])
                            st.divider()
    else:
        st.warning("No movie found with that name, please try again.")
