import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 1. DATA AND AI
# ==========================================

# Reading the files
movies = pd.read_csv('tmdb_5000_movies.zip', compression='zip')
credits = pd.read_csv('tmdb_5000_credits.zip', compression='zip')
credits.columns = ['movie_id', 'title', 'cast', 'crew']
df = movies.merge(credits, on='title')
# Regex function for JSON
def metin_temizle(metin):
    if pd.isna(metin):
        return ""
    bulunanlar = re.findall(r'"name":\s*"([^"]+)"', str(metin))
    return " ".join([i.lower().replace(" ", "") for i in bulunanlar])

df['temiz_turler'] = df['genres'].apply(metin_temizle)
df['temiz_anahtar_kelimeler'] = df['keywords'].apply(metin_temizle)
df['overview'] = df['overview'].fillna('')

df['bilgi_corbasi'] = df['temiz_turler'] + " " + df['temiz_anahtar_kelimeler'] + " " + df['overview'].str.lower()

# TF-IDF
tfidf_yeni = TfidfVectorizer(stop_words='english')
tfidf_matrisi_yeni = tfidf_yeni.fit_transform(df['bilgi_corbasi'])
benzerlik_matrisi_yeni = cosine_similarity(tfidf_matrisi_yeni, tfidf_matrisi_yeni)

# For quick search
film_indeksleri = pd.Series(df.index, index=df['title']).drop_duplicates()

# 6 types of reasons to enjoy watching a movie
sinema_boyutlari = {
    "Dramatik Yapı ve Duygu": ["drama", "emotion", "sad", "love", "family", "relationship", "tragedy", "crying", "feeling", "heart", "betrayal"],
    "Görsel ve İşitsel Estetik": ["cinematography", "visual", "soundtrack", "music", "color", "atmosphere", "beautiful", "effects", "aesthetic"],
    "Teknik Başarı ve Zanaat": ["director", "editing", "camera", "sound", "production", "oscar", "award", "masterpiece", "technique"],
    "Zihinsel Egzersiz ve Senaryo": ["twist", "mystery", "philosophical", "mind", "puzzle", "complex", "psychological", "metaphor", "intelligent"],
    "Escapizm (Gerçeklikten Kaçış)": ["sci-fi", "fantasy", "alien", "magic", "adventure", "action", "space", "future", "monster", "superhero"],
    "Sosyokültürel Keşif": ["history", "culture", "society", "biography", "war", "documentary", "politics", "world", "human", "period"]
}

# ==========================================
# 2. STREAMLIT 
# ==========================================

st.set_page_config(page_title="Gurme Sinema Öneri Sistemi", page_icon="🎬", layout="centered")

st.title("🎬 Yapay Zeka Tabanlı Gurme Sinema Asistanı")
st.write("İzlediğiniz bir filmi seçin ve o filmde sizi en çok etkileyen boyutu işaretleyin. Yapay zeka size nokta atışı gurme öneriler sunsun!")


film_listesi = sorted(df['title'].tolist())
secilen_film = st.selectbox("İzlediğiniz ve beğendiğiniz film:", film_listesi)


boyutlar = list(sinema_boyutlari.keys())
secilen_boyut = st.radio("Bu filmde en çok hangi yön ilginizi çekti?", boyutlar)


if st.button("Gurme Önerileri Getir 🎯"):
    if secilen_film in film_indeksleri:
        idx = film_indeksleri[secilen_film]
        boyut_kelimeleri = sinema_boyutlari[secilen_boyut]
        
        # Algorithm/tuple 
        skorlar = list(enumerate(benzerlik_matrisi_yeni[idx]))
        skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
        en_yakin_40_indeks = [i for i, skor in skorlar[1:41]]
        
        aday_filmler = df.iloc[en_yakin_40_indeks].copy()
        
        def boyut_puani_hesapla(bilgi_corbasi):
            puan = 0
            for kelime in boyut_kelimeleri:
                puan += bilgi_corbasi.count(kelime.lower())
            return puan

        aday_filmler['boyut_skoru'] = aday_filmler['bilgi_corbasi'].apply(boyut_puani_hesapla)
        sonuc = aday_filmler.sort_values(by='boyut_skoru', ascending=False).head(5)
        
        st.success(f"🍿 '{secilen_film}' filmini sevenler ve odağı [{secilen_boyut}] olanlar için yapay zeka sonuçları:")
        
        # Output
        for i, (_, satir) in enumerate(sonuc.iterrows(), 1):
            turler = satir['temiz_turler'].replace(" ", ", ")
            st.markdown(f"**{i}. {satir['title']}**")
            st.caption(f"🎥 Türler: {turler} | 🎯 Odak Skoru: {satir['boyut_skoru']}")
            st.divider()
