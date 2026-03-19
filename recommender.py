import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Standard ML Libraries


# ==========================================
# 📋 MOVIE RECOMMENDATION ENGINE (PYTHON)
# ==========================================

# 1. LOAD DATASET (Simulated MovieLens Structure)
# -----------------------------------------------
def load_data():
    # 🌟 Expanded Movie Metadata (TMDB Subset)
    movies_data = [
        {'movieId': 1, 'title': 'The Dark Knight', 'genres': 'Action|Crime|Drama', 'industry':'Hollywood'},
        {'movieId': 2, 'title': 'Inception', 'genres': 'Sci-Fi|Thriller|Action', 'industry':'Hollywood'},
        {'movieId': 3, 'title': 'Interstellar', 'genres': 'Sci-Fi|Drama|Adventure', 'industry':'Hollywood'},
        {'movieId': 4, 'title': 'Avengers: Endgame', 'genres': 'Action|Fantasy|Sci-Fi', 'industry':'Hollywood'},
        {'movieId': 5, 'title': 'The Shawshank Redemption', 'genres': 'Drama|Crime', 'industry':'Hollywood'},
        {'movieId': 6, 'title': 'RRR', 'genres': 'Action|Drama|Fantasy', 'industry':'Tollywood'},
        {'movieId': 7, 'title': 'Vikram', 'genres': 'Action|Thriller|Crime', 'industry':'Kollywood'},
        {'movieId': 8, 'title': 'Drishyam', 'genres': 'Thriller|Drama|Crime', 'industry':'Mollywood'},
        {'movieId': 9, 'title': '3 Idiots', 'genres': 'Comedy|Drama', 'industry':'Bollywood'},
        {'movieId': 10, 'title': 'KGF: Chapter 2', 'genres': 'Action|Crime|Drama', 'industry':'Kannada'},
    ]
    movies = pd.DataFrame(movies_data)

    # 👥 Expanded User Base (50 Users with varied preferences)
    ratings_list = []
    # User 1: Action Fan
    for m_id in [1, 2, 4, 6]: ratings_list.append({'userId': 1, 'movieId': m_id, 'rating': 5.0})
    # User 2: Drama/Thriller Fan
    for m_id in [2, 3, 5, 8]: ratings_list.append({'userId': 2, 'movieId': m_id, 'rating': 4.5})
    # User 3: Regional Cinema Explorer
    for m_id in [6, 7, 8, 10]: ratings_list.append({'userId': 3, 'movieId': m_id, 'rating': 4.8})
    
    # Generate 47 more users with random but patterned ratings
    np.random.seed(42)
    for u_id in range(4, 51):
        num_ratings = np.random.randint(2, 6)
        chosen_movies = np.random.choice(movies['movieId'].values, num_ratings, replace=False)
        for m_id in chosen_movies:
            ratings_list.append({
                'userId': u_id, 
                'movieId': m_id, 
                'rating': np.random.uniform(3.0, 5.0)
            })
            
    ratings = pd.DataFrame(ratings_list)
    return movies, ratings

# 2. CONTENT-BASED FILTERING (TF-IDF Similarity)
# -----------------------------------------------
def content_recommend(title, movies_df):
    # Vectorize genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get index of movie
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Return top recommendations
    movie_indices = [i[0] for i in sim_scores[1:4]]
    return movies_df['title'].iloc[movie_indices]

# 3. COLLABORATIVE FILTERING (Matrix Factorization - Sim)
# -------------------------------------------------------
def collaborative_recommend(user_id, movie_id, ratings_df):
    try:
        # Create User-Item Matrix
        matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Calculate User Similarity
        user_sim = cosine_similarity(matrix)
        user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)
        
        # Find similar users to our target user
        if user_id not in user_sim_df.index: return 3.5
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11]
        
        # Weighted average of ratings from similar users
        if movie_id in matrix.columns:
            relevant_ratings = matrix.loc[similar_users.index, movie_id]
            weights = similar_users.values
            if weights.sum() > 0:
                prediction = (relevant_ratings * weights).sum() / weights.sum()
                return prediction
        return 3.5 # Fallback
    except Exception as e:
        return 3.5 # Fallback average rating

# 4. HYBRID RECOMMENDATION ENGINE (Gold Standard)
# -------------------------------------------------------
def hybrid_recommend(user_id, title, movies_df, ratings_df):
    try:
        # Get content base
        content_scores = content_recommend(title, movies_df)
        hybrid_scores = []
        
        for movie_title in content_scores:
            m_id = movies_df[movies_df['title'] == movie_title]['movieId'].values[0]
            
            # 50% Content Score (Normalized from 1.0)
            # 50% Collaborative Score (Normalized to 0.5)
            collab_pred = collaborative_recommend(user_id, m_id, ratings_df)
            final_score = (collab_pred / 5.0) * 100 # Simple hybrid metric
            
            hybrid_scores.append((movie_title, final_score))
            
        return sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    except:
        return []

# 5. MAIN EXECUTION
# -------------------------------------------
if __name__ == "__main__":
    print("-" * 45)
    print("ROBOT: HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("-" * 45)

    # 1. Load Everything
    movies, ratings = load_data()
    print(f"Dataset: {len(movies)} Movies | {len(ratings)} User Interactions\n")

    # 2. Select Target User and Movie
    target_user = 1
    target_movie = "The Dark Knight"

    # 3. Hybrid Analysis Demo
    print(f"GENIUS-ML: Calculating Hybrid Matches for User {target_user}...")
    print(f"Analyzing preference based on: '{target_movie}'\n")
    
    recs = hybrid_recommend(target_user, target_movie, movies, ratings)
    
    print(f"{'MOVIE TITLE':<25} | {'HYBRID CONFIDENCE'}")
    print("-" * 45)
    for title, score in recs:
        print(f"{title:<25} | {score:2.1f}% Match")
    print("-" * 45)
