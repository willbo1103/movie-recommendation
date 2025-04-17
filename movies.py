from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')

def index():
    return render_template('index.html')

# ========== Load and preprocess dataset ==========
df = pd.read_csv('TMDB_movie_dataset_cleaned.csv')

# Handle missing values
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['genres'] = df['genres'].fillna('[]')
df['keywords'] = df['keywords'].fillna('[]')
df['original_language'] = df['original_language'].fillna('unknown')
df['vote_average'] = df['vote_average'].fillna(0)
df['vote_count'] = df['vote_count'].fillna(0)
df['adult'] = df['adult'].fillna(False)

# Normalize vote_count
scaler = MinMaxScaler()
df['vote_count_norm'] = scaler.fit_transform(df[['vote_count']])

# ========== Similarity component functions ==========
def rating_similarity(m1, m2):
    return 1 - abs(m1['vote_average'] - m2['vote_average']) / 10

def time_similarity(m1, m2):
    if pd.isna(m1['release_date']) or pd.isna(m2['release_date']):
        return 0.5
    delta = abs((m1['release_date'] - m2['release_date']).days)
    return max(0, 1 - delta / 3650)

def adult_similarity(m1, m2):
    return 1 if m1['adult'] == m2['adult'] else 0

def vote_count_similarity(m1, m2):
    return 1 - abs(m1['vote_count_norm'] - m2['vote_count_norm'])

def genre_similarity(m1, m2):
    def to_set(s):
        if pd.isna(s):
            return set()
        return set([kw.strip().lower() for kw in str(s).split(',')])
    g1 = to_set(m1['genres'])
    g2 = to_set(m2['genres'])
    if not g1 or not g2:
        return 0
    return len(g1 & g2) / len(g1)

def keyword_similarity(m1, m2):
    def to_set(s):
        if pd.isna(s):
            return set()
        return set([kw.strip().lower() for kw in str(s).split(',')])
    k1 = to_set(m1['keywords'])
    k2 = to_set(m2['keywords'])
    if not k1 or not k2:
        return 0
    return len(k1 & k2) / len(k1)

def language_similarity(m1, m2, preferred_languages):
    if m1['original_language'] not in preferred_languages or m2['original_language'] not in preferred_languages:
        return 0
    return 1 if m1['original_language'] == m2['original_language'] else 0.5

def total_similarity(m1, m2, preferred_languages=['en']):
    weights = {
        'rating': 150.0,
        'time': 25.0,
        'adult': 50.0,
        'vote_count': 50.0,
        'genres': 150.0,
        'language': 50.0,
        'keywords': 1000.0
    }
    score = 0
    score += weights['rating'] * rating_similarity(m1, m2)
    score += weights['time'] * time_similarity(m1, m2)
    score += weights['adult'] * adult_similarity(m1, m2)
    score += weights['vote_count'] * vote_count_similarity(m1, m2)
    score += weights['genres'] * genre_similarity(m1, m2)
    score += weights['language'] * language_similarity(m1, m2, preferred_languages)
    score += weights['keywords'] * keyword_similarity(m1, m2)
    return score

# ========== Recommendation by movie title ==========
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_ids = data.get('ids', [])  # <- matches frontend JSON
    preferred_languages = data.get('preferred_languages', ['en'])

    target_movies = df[df['id'].astype(int).isin(movie_ids)]
    if target_movies.empty:
        return jsonify({'error': 'No matching movie IDs found in dataset.'}), 400

    similarities = []
    for idx, row in df.iterrows():
        if int(row['id']) in movie_ids:
            continue
        total_score = sum(
            total_similarity(target, row, preferred_languages)
            for _, target in target_movies.iterrows()
        )
        avg_score = total_score / len(target_movies)
        similarities.append({
            'id': int(row['id']),
            'title': row['title'],
            'release_date': row['release_date'].strftime('%Y-%m-%d') if not pd.isna(row['release_date']) else 'Unknown',
            'poster_path': row['poster_path'],  # ✅ Include poster path
            'genres': row['genres'],  # ✅ Include genres in the recommendation
            'overview': row['overview'],
            'backdrop_path': row['backdrop_path'],
            'original_language': row['original_language'],
            'score': round(avg_score, 4)
        })

    # ✅ Sort all by score without slicing top 10
    sorted_recs = sorted(similarities, key=lambda x: x['score'], reverse=True)

    return jsonify({'recommendations': sorted_recs})

# ========== Endpoint for dropdown options ==========
@app.route('/titles', methods=['GET'])
def get_titles():
    titles = df[['id', 'title']].drop_duplicates().to_dict(orient='records')
    return jsonify(titles)

if __name__ == '__main__':
    app.run(debug=True)
