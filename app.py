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
# 1. LIVE DATA RECRUITMENT POOL
# ==========================================
def fetch_recommendation_pool(movie_id):
    """
    Fetches similar, top-rated, and popular movies from TMDB API 
    real-time to dynamically train our AI model.
    """
    api_key = st.secrets["TMDB_API_KEY"]
    pool = []
    seen_ids = set()
    
    # Gathering movie candidates from 4 different API streams
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
                        'poster_path': m.get('poster_path', '')
                    })
        except:
            continue
            
    return pd.DataFrame(pool)


movie_indices = pd.Series(df.index, index=df['title'])

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

search_query = st.text_input("Select a movie you watched and liked (e.g., The Love Witch, Saw):", "")

if search_query:
    api_key = st.secrets["TMDB_API_KEY"]
    search_url = f"https://themoviedb.org{api_key}&query={search_query}&language=en-US"
    search_results = requests.get(search_url).json().get('results', [])
    
    if search_results:
        movie_options = {f"{m['title']} ({m.get('release_date', '')[:4]})": m for m in search_results}
        selected_display = st.selectbox("Which movie do you mean exactly?", list(movie_options.keys()))
        target_movie = movie_options[selected_display]

dimensions = list(cinema_dimensions.keys())
selected_dimension = st.radio("Which aspect of this movie captivated you the most?", dimensions)

if st.button("Discover Gourmet Matches"):
           
            df_pool = fetch_recommendation_pool(target_movie['id'])
            
            if not df_pool.empty:
            
                orig_row = pd.DataFrame([{
                    'movie_id': target_movie['id'],
                    'title': target_movie['title'],
                    'overview': target_movie['overview'],
                    'vote_average': target_movie.get('vote_average', 0.0),
                    'poster_path': target_movie.get('poster_path', ''),
                    'clean_genres': ""
                }])
                df = pd.concat([orig_row, df_pool]).drop_duplicates(subset=['movie_id']).reset_index(drop=True)
                
                # --- REAL-TIME AI PIPELINE (TF-IDF & COSINE SIMILARITY) ---
                tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf_vectorizer.fit_transform(df['overview'].str.lower())
                similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                idx = 0 
        dimension_words = cinema_dimensions[selected_dimension]
        
        selected_row = df.iloc[idx]
        selected_title_lower = selected_movie.lower()
        selected_overview_clean = selected_row['overview'].strip().lower()
        
        selected_title_words = set([w for w in selected_title_lower.split() if len(w) > 2])
        
        selected_cast = str(selected_row.get('cast', '')).lower()
        selected_crew = str(selected_row.get('crew', '')).lower()
        
        selected_cast_words = set(re.findall(r'"name":\s*"([^"]+)"', selected_cast)[:3])
        selected_cast_words = {w.lower() for name in selected_cast_words for w in name.split()}
        
        selected_director = ""
        director_match = re.search(r'"job":\s*"Director",\s*"name":\s*"([^"]+)"', selected_crew)
        if director_match:
            selected_director = director_match.group(1).lower()

       
        scores = list(enumerate(similarity_matrix[idx]))
        
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_150_candidates = scores[1:151]
        
        valid_indices = []
        tfidf_scores_map = {}
        seen_overviews = set()
        
        if selected_overview_clean:
            seen_overviews.add(selected_overview_clean)
            
        for i, score in top_150_candidates:
            candidate_row = df.iloc[i]
            candidate_title = candidate_row['title'].lower()
            candidate_overview = candidate_row['overview'].strip().lower()
            
          
            if candidate_overview in seen_overviews or any(candidate_overview[:30] == seen[:30] for seen in seen_overviews):
                continue
                
        
            candidate_title_words = set([w for w in candidate_title.split() if len(w) > 2])
            is_same_name = bool(selected_title_words.intersection(candidate_title_words))
            
            candidate_cast = str(candidate_row.get('cast', '')).lower()
            candidate_crew = str(candidate_row.get('crew', '')).lower()
            
            candidate_director = ""
            c_dir_match = re.search(r'"job":\s*"Director",\s*"name":\s*"([^"]+)"', candidate_crew)
            if c_dir_match:
                candidate_director = c_dir_match.group(1).lower()
                
            candidate_cast_names = set(re.findall(r'"name":\s*"([^"]+)"', candidate_cast)[:3])
            candidate_cast_words = {w.lower() for name in candidate_cast_names for w in name.split()}
            
            is_same_director = (selected_director == candidate_director) if selected_director else False
            is_same_actors = bool(selected_cast_words.intersection(candidate_cast_words)) if selected_cast_words else False
            
            if is_same_name or (is_same_director and is_same_actors):
                continue
                
            valid_indices.append(i)
            tfidf_scores_map[i] = score
            if candidate_overview:
                seen_overviews.add(candidate_overview)
                
            if len(valid_indices) >= 40:
                break
                
     
        candidate_movies = df.iloc[valid_indices].copy().reset_index(drop=True)
        
        def calculate_dimension_score(info_soup):
            score = 0
            for word in dimension_words:
                score += len(re.findall(r'\b' + re.escape(word.lower()) + r'\b', info_soup))
            return score
            
        candidate_movies['dimension_score'] = candidate_movies['info_soup'].apply(calculate_dimension_score)
        
       
        candidate_movies['tfidf_score'] = candidate_movies['title'].map({df.loc[k, 'title']: s for k, s in tfidf_scores_map.items()})
      
        def apply_imdb_penalty(row):
            if row['vote_average'] < 6.0:
                return float(row['dimension_score']) * 0.5
            return float(row['dimension_score'])

        candidate_movies['final_dimension_score'] = candidate_movies.apply(apply_imdb_penalty, axis=1)
        
        # Calculate the Score
        candidate_movies['quality_score'] = (candidate_movies['tfidf_score'] * 0.4) + ((candidate_movies['vote_average'] / 10.0) * 0.6)

        results = candidate_movies.sort_values(by=['final_dimension_score', 'quality_score'], ascending=[False, False]).head(5)

        
        # ==========================================
        # 3. DISPLAY RESULTS
        # ==========================================
        with st.container():
            st.success(f"'{target_movie['title']}' AI results for those who love [{selected_dimension}]:")

            
            for i, (_, row) in enumerate(results.iterrows(), 1):
                genres_list = ", ".join(row['clean_genres'].split()).title() if row['clean_genres'] else "Not Specified"
                
                tf_val = row['tfidf_score']
                similarity_percentage = round(tf_val * 100, 1) if pd.notna(tf_val) else 0.0
                
                
                col1, col2 = st.columns([1, 2.3])
                
                with col1:
                    poster_url = get_poster_url(row['movie_id'])
                    st.image(poster_url, use_container_width=True)
                    
                with col2:
         
                    st.markdown(f"### {i}. {row['title']}")
                    imdb_score = row.get('vote_average', 0.0)
                    st.markdown(f"⭐ **IMDb:** `{imdb_score}/10`  |  **Genres:** {genres_list}")
                    
                    if row['vote_average'] < 6.0:
                        st.write(f" Match Score: `{int(row['dimension_score'])}` (Cezalı: `{row['final_dimension_score']}`)")
                    else:
                        st.write(f" Match Score: `{int(row['dimension_score'])}`")
                        
                    st.write(" AI Similarity Match:")
                    
                    progress_value = float(tf_val) if pd.notna(tf_val) and tf_val <= 1.0 else 0.0
                    st.progress(progress_value)
                    st.caption(f"Match Rate: %{similarity_percentage}")
                    
                
                    movie_overview = row.get('overview', '').strip()
                    if movie_overview:
                        with st.expander("Read Overview"):
                            st.write(movie_overview)
    
                st.divider()
                    else:
        st.warning("No movie found with that name, please try again.")

                    
