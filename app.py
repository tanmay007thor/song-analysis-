from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
data = pd.read_csv('./data/preprocess.csv')

# Load the pre-trained model
autoencoder = tf.keras.models.load_model('./model/song_autoencoder_model.h5', compile=False)
encoder = tf.keras.models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_6').output)

# Features used for encoding
features = [
    'variance', 'Tempo', 'Loudness', 'Explicit', 'Popularity', 'Energy',
    'Danceability', 'Positiveness', 'Speechiness', 'Liveness',
    'Acousticness', 'Instrumentalness'
]

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(data[features])

# Recommendation function
def get_song_recommendations(song_name, top_n=20):
    song_name = song_name.strip().lower()  # Normalize input
    data['normalized_song_name'] = data['song name'].str.strip().str.lower()

    if song_name not in data['normalized_song_name'].values:
        raise ValueError(f"Song '{song_name}' not found in the dataset.")

    song_index = data[data['normalized_song_name'] == song_name].index[0]
    song_encoded = encoder.predict(X[song_index].reshape(1, -1))
    encoded_features = encoder.predict(X)

    similarities = cosine_similarity(song_encoded, encoded_features)
    similar_song_indices = similarities[0].argsort()[-top_n-1:-1][::-1]

    similar_songs = data.iloc[similar_song_indices][['song name', 'artist name']]
    similar_songs['Similarity'] = similarities[0][similar_song_indices]

    return similar_songs

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error = None

    if request.method == 'POST':
        song_name = request.form['song_name']

        try:
            recommendations = get_song_recommendations(song_name)
        except ValueError as e:
            error = str(e)

    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    app.run(debug=True)
