# 🎬 Movie Recommender

A full-stack movie recommendation system that suggests similar movies using 
TF-IDF content-based filtering, combined with live movie data (posters, details, 
trending, popular, top-rated) fetched from the TMDB API.

🔗 **Live Demo:** [movie-recommendation-system001.streamlit.app](https://movie-recommendation-system001.streamlit.app/)

---

## ✨ Features

- 🔍 **Search** — search movies by title with live suggestions
- 🏠 **Home Feed** — browse trending, popular, top-rated, now-playing, and upcoming movies
- 📄 **Movie Details** — overview, genres, release date, poster, and backdrop
- 🤖 **TF-IDF Recommendations** — content-based similar movie suggestions from a trained ML model
- 🎭 **Fallback Recommendations** — genre-based similar movies (via TMDB) when a movie isn't in the trained dataset
- ⚡ **Fast & Cached** — API responses cached for smoother browsing

---

## 🛠️ Tech Stack

| Layer      | Technology                          |
|------------|--------------------------------------|
| Frontend   | Streamlit                            |
| Backend    | FastAPI                              |
| ML Model   | TF-IDF Vectorization (scikit-learn)  |
| Data       | TMDB API + trained dataset (`.pkl`)  |
| Deployment | Render (backend) + Streamlit Community Cloud (frontend) |

---

## 🌐 Deployment

- **Backend** is deployed on [Render](https://render.com)
- **Frontend** is deployed on [Streamlit Community Cloud](https://share.streamlit.io)

---

## 📷 Screenshots
<img width="1919" height="977" alt="Screenshot 2026-07-26 151837" src="https://github.com/user-attachments/assets/afa836fd-4217-4fa9-bd14-e931319b6667" />


<img width="1919" height="958" alt="Screenshot 2026-07-26 151944" src="https://github.com/user-attachments/assets/84fd92df-5d8a-40a6-9361-f28e1a320483" />


<img width="1919" height="968" alt="Screenshot 2026-07-26 152044" src="https://github.com/user-attachments/assets/0b84df9b-8c4e-403c-a630-a48e8e7b1c21" />
