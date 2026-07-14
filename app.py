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
# 2. STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Gourmet Movie Recommender System", page_icon="🎬", layout="centered")
st.title("AI-Powered Cinema Assistant")
st.write("Select a movie you have watched and choose the aspect that impressed you the most. Let AI provide you with pinpoint gourmet recommendations!")


search_query = st.text_input("Bir film adı yazın (Örn: The Love Witch, Saw):", "")

if search_query:
    api_key = st.secrets["TMDB_API_KEY"]
    search_url = f"https://themoviedb.org{api_key}&query={search_query}&language=en-US"
    
    try:
        search_results = requests.get(search_url, timeout=3).json().get('results', [])
    except:
        search_results = []
    
    if search_results:
        movie_options = {f"{m['title']} ({m.get('release_date', '')[:4]})": m for m in search_results}
        selected_display = st.selectbox("Eşleşen filmler arasından tam olarak hangisi?", list(movie_options.keys()))
        target_movie = movie_options[selected_display]

dimensions = list(cinema_dimensions.keys())
selected_dimension = st.radio("Which aspect of this movie captivated you the most?", dimensions)


        if st.button("Discover Gourmet Matches"):
            with st.spinner("Gurme Yapay Zeka modeli dinamik veri setini eğitiyor..."):
                
                
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
                    df_pool = pd.concat([orig_row, df_pool]).drop_duplicates(subset=['movie_id']).reset_index(drop=True)
                    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = tfidf_vectorizer.fit_transform(df_pool['overview'].str.lower())
                    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
              
                    scores = list(enumerate(similarity_matrix[0]))
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)
                    top_candidates = scores[1:]
                    
                    valid_indices = []
                    tfidf_scores_map = {}
                    for idx, score in top_candidates:
                        valid_indices.append(idx)
                        tfidf_scores_map[idx] = score
                        
                    candidate_movies = df_pool.iloc[valid_indices].copy().reset_index(drop=True)
                    
              
                    def calculate_dimension_score(overview):
                        score = 0
                        for word in dimension_words:
                            score += len(re.findall(r'\b' + re.escape(word.lower()) + r'\b', str(overview).lower()))
                        return score
                        
                    candidate_movies['dimension_score'] = candidate_movies['overview'].apply(calculate_dimension_score)
                    candidate_movies['tfidf_score'] = candidate_movies['title'].map({df_pool.loc[k, 'title']: s for k, s in tfidf_scores_map.items()})
                    
                  
                    def apply_imdb_penalty(row):
                        if row['vote_average'] < 6.0:
                            return float(row['dimension_score']) * 0.5
                        return float(row['dimension_score'])
                    candidate_movies['final_dimension_score'] = candidate_movies.apply(apply_imdb_penalty, axis=1)
                    
                 
                    candidate_movies['quality_score'] = (candidate_movies['tfidf_score'] * 0.4) + ((candidate_movies['vote_average'] / 10.0) * 0.6)
                    results = candidate_movies.sort_values(by=['tfidf_score', 'quality_score'], ascending=[False, False]).head(5)
                    
                
                    with st.container():
                        st.success(f"'{target_movie['title']}' AI results for those who love [{selected_dimension}]:")
                        
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
                                    st.write(f"Match Score: `{int(row['dimension_score'])}` (Cezalı: `{row['final_dimension_score']}`)")
                                else:
                                    st.write(f" Match Score: `{int(row['dimension_score'])}`")
                                    
                                st.write("AI Similarity Match:")
                                progress_value = float(tf_val) if pd.notna(tf_val) and tf_val <= 1.0 else 0.0
                                st.progress(progress_value)
                                st.caption(f"Match Rate: %{similarity_percentage}")
                                
                                if row['overview']:
                                    with st.expander("Base Overview"):
                                        st.write(row['overview'])
                            st.divider()

        
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
            st.success(f"🍿 '{selected_movie}' AI results for those who love [{selected_dimension}] and focus on:")
            
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
                    



