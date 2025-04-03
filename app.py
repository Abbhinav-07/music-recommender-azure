from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # Enable CORS for local frontend requests

# Load dataset
file_path = "data/data.csv"
df = pd.read_csv(file_path)

df = df[df["popularity"] > 65]  # Filter high popularity songs
df["name_lower"] = df["name"].str.lower()  # Case-insensitive search

# Load trained model
hdbscan_model_path = "models/gmm_model.pkl"
try:
    loaded_hdbscan_model = joblib.load(hdbscan_model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    loaded_hdbscan_model = None

# Standardize features
features = ["valence", "acousticness", "danceability", "energy",
            "instrumentalness", "liveness", "loudness", "speechiness", "tempo"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

if loaded_hdbscan_model:
    df["cluster"] = loaded_hdbscan_model.fit_predict(X_scaled)

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "Server is running locally 🚀"}), 200

@app.route("/recommend", methods=["GET"])
def recommend_songs():
    if not loaded_hdbscan_model:
        return jsonify({"error": "Model not loaded."}), 500

    track_name = request.args.get("track_name", "").strip().lower()
    year = request.args.get("year", type=int)

    if not track_name or not year:
        return jsonify({"error": "Provide both 'track_name' and 'year'."}), 400

    track_row = df[(df["name_lower"] == track_name) & (df["year"] == year)]
    if track_row.empty:
        return jsonify({"error": "Song not found. Check the name and year."}), 404

    track_cluster = track_row.iloc[0]["cluster"]
    
    if track_cluster == -1:
        return jsonify({"message": "This song is classified as an outlier."})

    similar_songs = df[(df["cluster"] == track_cluster) & (df["name_lower"] != track_name)]
    recommendations = similar_songs.sample(min(5, len(similar_songs)))[["name", "artists", "year"]].to_dict(orient="records")

    return jsonify({"original_song": {"name": track_name, "year": year}, "similar_songs": recommendations})

if __name__ == "__main__":
    print("🚀 Running Flask server locally...")
    app.run(host="0.0.0.0", port=5000, debug=True)  # Localhost with port 5000
