import streamlit as st
import requests
import re

# ==========================================
# 0. CONFIG & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gourmet Movie Recommender System", page_icon="🎬", layout="centered")

# TMDB API 
if "TMDB_API_KEY" in st.secrets:
    API_KEY = st.secrets["TMDB_API_KEY"]
else:
    st.error("Lütfen Streamlit secrets alanına 'TMDB_API_KEY' değerini ekleyin.")
    st.stop()

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

cinema_dimensions = {
    "Story & Intense Emotion": ["drama", "emotion", "sad", "love", "family", "relationship", "tragedy", "crying", "feeling", "heart", "betrayal", "romance", "melodrama"],
    "Cinematography & Atmosphere": ["cinematography", "visual", "soundtrack", "music", "color", "atmosphere", "beautiful", "effects", "aesthetic", "neon", "stylized"],
    "Direction & Craft": ["director", "editing", "camera", "sound", "production", "oscar", "award", "masterpiece", "technique", "art", "indie"],
    "Mind Games & Plot Twists": ["twist", "mystery", "philosophical", "mind", "puzzle", "complex", "psychological", "metaphor", "intelligent", "existential", "thriller"],
    "Fantasy Worlds & Escapism": ["sci-fi", "fantasy", "alien", "magic", "adventure", "action", "space", "future", "monster", "superhero", "cyberpunk"],
    "Period & Real History": ["history", "culture", "society", "biography", "war", "documentary", "politics", "world", "human", "period", "historical"]
}

# ==========================================
# 1. HELPER FUNCTIONS (TMDB LIVE CALLS)
# ==========================================

def search_movie_live(query):
    """Kullanıcının yazdığı kelimeye göre TMDB'de canlı film arar (İlk 10 sonucu döner)"""
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": query, "language": "en-US", "page": 1}
    try:
        response = requests.get(url, params=params, timeout=3).json()
        return response.get("results", [])
    except Exception:
        return []

def get_movie_details_and_keywords(movie_id):
    """Bir filmin detaylarını ve keyword'lerini çekip 'info_soup' (çorba metin) oluşturur"""
    details_url = f"{BASE_URL}/movie/{movie_id}"
    keywords_url = f"{BASE_URL}/movie/{movie_id}/keywords"
    params = {"api_key": API_KEY}
    
    info_soup = ""
    genres = []
    
    try:
        # Details
        det_resp = requests.get(details_url, params=params, timeout=3).json()
        overview = det_resp.get("overview", "").lower()
        genres = [g["name"] for g in det_resp.get("genres", [])]
        genres_clean = " ".join([g.lower().replace(" ", "") for g in genres])
        
        # Keyword
        key_resp = requests.get(keywords_url, params=params, timeout=3).json()
        keywords = [k["name"] for k in key_resp.get("keywords", [])]
        keywords_clean = " ".join([k.lower().replace(" ", "") for k in keywords])
        
        info_soup = f"{genres_clean} {keywords_clean} {overview}"
    except Exception:
        pass
        
    return info_soup, genres

def get_live_recommendations(movie_id):
    """TMDB Recommendations API kullanarak doğrudan benzer 20 filmi çeker"""
    url = f"{BASE_URL}/movie/{movie_id}/recommendations"
    params = {"api_key": API_KEY, "language": "en-US", "page": 1}
    try:
        response = requests.get(url, params=params, timeout=3).json()
        return response.get("results", [])
    except Exception:
        return []

# ==========================================
# 2. STREAMLIT INTERFACE
# ==========================================
st.title("AI-Powered Gourmet Cinema Assistant")
st.write("Select a movie you have watched and choose the aspect that impressed you the most. Let AI provide you with pinpoint gourmet recommendations!")


search_query = st.text_input("Movie Name:", placeholder="Inception, Interstellar, Dune...")

selected_movie_data = None

if search_query:
    search_results = search_movie_live(search_query)
    
    if search_results:
  
        movie_options = []
        movie_map = {}
        
        for m in search_results:
            release_year = m.get("release_date", "N/A")[:4] if m.get("release_date") else "N/A"
            label = f"{m['title']} ({release_year})"
            movie_options.append(label)
            movie_map[label] = m
            
        selected_label = st.selectbox("Choose the right mpvie:", movie_options)
        selected_movie_data = movie_map[selected_label]
    else:
        st.warning("Warning no connection")

dimensions = list(cinema_dimensions.keys())
selected_dimension = st.radio("What was the most impressing thing about the movie?", dimensions)


if st.button("Discover Gourmet Matches") and selected_movie_data:
    target_id = selected_movie_data["id"]
    target_title = selected_movie_data["title"]
    dimension_words = cinema_dimensions[selected_dimension]
    
    with st.spinner("Loading..."):
      
        raw_recs = get_live_recommendations(target_id)
        
        if not raw_recs:
            st.warning("No recommendations for this movie.")
        else:
            scored_candidates = []
            
            for rec in raw_recs[:20]: 
                rec_id = rec["id"]
                rec_title = rec["title"]
                vote_avg = rec.get("vote_average", 0.0)
   
                soup, genres = get_movie_details_and_keywords(rec_id)
                
           
                dim_score = 0
                for word in dimension_words:
                    dim_score += len(re.findall(r'\b' + re.escape(word.lower()) + r'\b', soup))
                    
            
                final_dim_score = dim_score * 0.5 if vote_avg < 6.0 else float(dim_score)
                
            
                quality_score = (vote_avg / 10.0) * 0.6 + (rec.get("popularity", 0.0) / 500.0) * 0.4
                
                scored_candidates.append({
                    "title": rec_title,
                    "overview": rec.get("overview", "No overview available."),
                    "poster_path": rec.get("poster_path"),
                    "vote_average": vote_avg,
                    "genres": genres,
                    "dim_score": dim_score,
                    "final_dim_score": final_dim_score,
                    "quality_score": quality_score
                })
            
        
            results = sorted(scored_candidates, key=lambda x: (x["final_dim_score"], x["quality_score"]), reverse=True)[:5]
            
            st.success(f"🍿 [{selected_dimension}] odaklı '{target_title}' benzeri gurme seçimlerimiz:")
            st.divider()
            
            for i, movie in enumerate(results, 1):
                col1, col2 = st.columns([1, 2.3])
                
                with col1:
                    if movie["poster_path"]:
                        poster_url = f"{IMAGE_BASE_URL}{movie['poster_path']}"
                    else:
                        poster_url = "https://via.placeholder.com/500x750?text=No+Poster"
                    st.image(poster_url, use_container_width=True)
                    
                with col2:
                    st.markdown(f"### {i}. {movie['title']}")
                    genres_list = ", ".join(movie["genres"]) if movie["genres"] else "Not Specified"
                    st.markdown(f"⭐ **IMDb:** `{movie['vote_average']:.1f}/10`  |  **Genres:** {genres_list}")
                    
                    if movie['vote_average'] < 6.0:
                        st.write(f"**Dimension Match Score:** `{int(movie['dim_score'])}` (Cezalı: `{movie['final_dim_score']}`)")
                    else:
                        st.write(f"**Dimension Match Score:** `{int(movie['dim_score'])}`")
                    
                    if movie["overview"]:
                        with st.expander("Read Overview"):
                            st.write(movie["overview"])
                
                st.divider()
