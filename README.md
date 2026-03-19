# 🎬 CineWorld – Global Movie & TV Discovery

CineWorld is a high-performance, real-time web application for discovering millions of movies and TV shows from around the world. Powered by the **TMDB (The Movie Database) API** and integrated with the **YouTube Data API**, it provides a seamless experience for exploring trailers, cast details, streaming availability, and more.

## 🚀 Key Features

- **🌍 500,000+ Database**: Access the entire TMDB library, including Hollywood, Bollywood, Tollywood, and international cinema.
- **📺 Web Series Support**: Dedicated sections for trending and popular TV shows/web series.
- **🎬 Smart Trailer Playback**: 
  - Integrated **YouTube API** fallback to ensure trailers always work.
  - Custom embed parameters to fix common cross-origin playback errors (Error 153/301).
- **🎭 Regional Excellence**: Specialized rows for Indian regional cinema (Hindi, Telugu, Tamil, Malayalam, Kannada) and Global cinema (Korean, Japanese, French, Spanish).
- **🤖 CineMatch AI (Recommendation Engine)**: A built-in ML logic to find similar movies based on genre, industry, era, and rating.
- **📊 Real-Time Analytics**: Visual charts showing database statistics for different industries and languages.

## 🛠️ Setup & Installation

Since CineWorld is a pure frontend application, it requires no complex installation. 

1. **Get your API Keys**:
   - **TMDB Key**: Register at [themoviedb.org](https://www.themoviedb.org/documentation/api) to get your free API key.
   - **YouTube Key (Optional)**: Generate a key at [Google Cloud Console](https://console.cloud.google.com/) for more resilient trailer search.
2. **Launch the App**:
   - Open `index.html` in any modern browser (Chrome, Edge, etc.).
   - Enter your API keys on the launch screen.

- `index.html`: The core structure and layout.
- `style.css`: Modern, dark-themed styling with responsive design.
- `app.js`: Main application logic for real-time TMDB fetching.
- `script.js`: The browser-side ML recommendation engine for instant similarity scores.
- `recommender.py`: The core Python ML engine implementing SVD-based Collaborative Filtering.
- `ML_REPORT.md`: Formal project documentation for the NRCM Machine Learning initiative.

## 🐍 Python Recommendation Engine

The project now includes a standalone Python implementation (`recommender.py`) for advanced model training:

1.  **Content-Based**: Uses `TF-IDF` and `Cosine Similarity`.
2.  **Collaborative Filtering**: Uses the `Surprise` library's `SVD` (Singular Value Decomposition) algorithm.
3.  **Hybrid Modeling**: Ready to merge user-rating predictions with content attributes.

**To run the Python core:**
```bash
pip install -r requirements.txt
python recommender.py
```

## 📈 ML Project Report (NRCM)

A comprehensive project report is available in [ML_REPORT.md](./ML_REPORT.md). It details:
- **Project Objectives**: Why machine learning is vital for recommendation systems.
- **Methodology**: The 6-step pipeline from data collection to prediction.
- **Algorithm Deep-Dive**: Explanations for Content, Collaborative, and Hybrid systems.
- **Tools**: List of libraries used (Pandas, Scikit-learn, Surprise).

## 📤 Future Deployment (Commands)
...

To push your latest changes to GitHub:

```powershell
git add .
git commit -m "Add Web Series, YouTube API integration, and 500k+ movie database rows"
git push origin main
```

---
*Created for the CineWorld Project.*
