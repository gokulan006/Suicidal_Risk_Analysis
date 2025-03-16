from flask import Flask, jsonify, request, render_template
import praw
import pandas as pd
import re
import os
from nltk.corpus import stopwords
import nltk
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from geopy.geocoders import Nominatim
from flask_sqlalchemy import SQLAlchemy
import folium
from folium.plugins import HeatMap
from sqlalchemy import create_engine
from dash_app import create_dashboard
import torch
from sentence_transformers import SentenceTransformer, util

# Load pre-trained BERT model
bert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
high_risk_keywords = [
    "suicide", "hopeless", "self-harm", "die", "kill", "worthless", "end it", "pain", "depressed",
    "overdose", "no future", "give up", "alone", "empty", "dark", "nothing matters", "can't go on",
    "useless", "suffering", "painless death"
]

moderate_risk_keywords = [
    "sad", "anxious", "stress", "struggling", "breakdown", "crying", "overwhelmed", "tired",
    "panic", "fear", "low", "nobody cares", "helpless", "isolated", "insecure"
]

low_risk_keywords = [
    "mental health", "therapy", "healing", "self-care", "meditation", "help", "wellness",
    "recovery", "coping", "mindfulness", "support", "stay strong", "hope"
]

# Compute embeddings for each category
high_risk_embeds = bert_model.encode(high_risk_keywords, convert_to_tensor=True)
moderate_risk_embeds = bert_model.encode(moderate_risk_keywords, convert_to_tensor=True)
low_risk_embeds = bert_model.encode(low_risk_keywords, convert_to_tensor=True)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'reddit_posts.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define the RedditPost model
class RedditPost(db.Model):
    __tablename__ = "reddit_posts"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    subreddit = db.Column(db.String(100))
    post_id = db.Column(db.String(50), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime)
    title = db.Column(db.Text)
    content = db.Column(db.Text)
    upvotes = db.Column(db.Integer)
    comments = db.Column(db.Integer)
    url = db.Column(db.Text)
    sentiment = db.Column(db.String(50))
    risk_level = db.Column(db.String(50))

# Create the database and table
with app.app_context():
    db.create_all()

dash_app = create_dashboard(app, db)

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="uxYCp1MHk4lcSW4IuRt_Ww",
    client_secret="qYuom6J37bzRN9toPiqQ21aJPitLFQ",
    user_agent="CrisisTracker/1.0",
    username="Accomplished-Oil9148",
    password="gokulan06",
    check_for_async=False,
    verify=False
)

# Load Sentiment Analyzer
analyser = SentimentIntensityAnalyzer()

# Load NLP model for location extraction
nlp = spacy.load("en_core_web_sm")

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Initialize geocoder
geolocator = Nominatim(user_agent="crisis_mapping")

# Load risk prediction model and tokenizer
model = keras.models.load_model("bigru_mha_cnn_model.keras")
with open("bigru_mha_cnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

risk_labels = ["Moderate Risk", "Low Risk", "High Risk"]

# Define subreddits and keywords
subreddits = ["mentalhealth", "depression", "SuicideWatch", "Anxiety", "addiction"]
keywords = [
    "anxiety", "depression", "panic attack", "mental breakdown", "hopeless", "exhausted", "trauma", "PTSD", "insomnia", "overwhelmed",
    "suicidal", "want to die", "end it all", "no way out", "I can’t do this anymore", "self-harm", "cutting", "relapse", "broken inside", "feeling empty",
    "addiction help", "withdrawal", "overdose", "rehab", "relapse", "alcoholic", "drunk again", "high again", "need drugs", "can't stop using",
    "I need help", "feeling lost", "therapy isn’t working", "someone talk to me", "help me please", "nobody understands", "can’t keep going", "I'm scared", "no one cares", "I feel alone",
    "lonely", "no friends", "worthless", "useless", "can't sleep", "tired of everything", "crying all night", "can’t eat", "numb inside", "want to disappear"
]

country_mapping = {
    'US': 'United States', 'U.S.': 'United States', 'USA': 'United States', 'U.S': 'United States',
    'IN': 'India', 'CA': 'Canada', 'Italy': 'Italy', 'UK': 'United Kingdom', 'U.K.': 'United Kingdom',
    'Philippines': 'Philippines', 'France': 'France', 'Spain': 'Spain', 'Mexico': 'Mexico',
    'Germany': 'Germany', 'Berlin': 'Germany', 'China': 'China', 'Denmark': 'Denmark',
    'Turkey': 'Turkey', 'Sweden': 'Sweden', 'Iraq': 'Iraq', 'Netherlands': 'Netherlands',
    'Puerto Rico': 'Puerto Rico', 'NZ': 'New Zealand', 'Norway': 'Norway'
}

# Function to extract posts
def extract_reddit_posts(limit=50):
    posts_data = []

    try:
        for subreddit_name in subreddits:
            print(f"Fetching posts from r/{subreddit_name}...")
            subreddit = reddit.subreddit(subreddit_name)

            for post in subreddit.new(limit=limit):  # Fetch latest posts
                if any(keyword.lower() in post.title.lower() + post.selftext.lower() for keyword in keywords):
                    posts_data.append([
                        subreddit_name,  # Subreddit name
                        post.id,
                        post.created_utc,
                        post.title,
                        post.selftext,
                        post.score,
                        post.num_comments,
                        post.url
                    ])

        # Convert to DataFrame
        columns = ["Subreddit", "Post ID", "Timestamp", "Title", "Content", "Upvotes", "Comments", "URL"]
        df = pd.DataFrame(posts_data, columns=columns)

        return df
    except Exception as e:
        print(f"An error occurred: {e}")

def clean_text(text):
    """ Cleans text by removing special characters, stopwords, and emojis """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # Remove stopwords
    return text

# Function to analyze sentiment
def sentiment_scores(sentence):
    if isinstance(sentence, float):
        sentence = str(sentence)
    sentiment_dict = analyser.polarity_scores(sentence)
     
    if sentiment_dict['compound'] >= 0.10:
        return "Positive"
    elif sentiment_dict['compound'] <= -0.10:
        return "Negative"
    else:
        return "Neutral"
def classify_risk_level(user_text):
    """Classifies risk level using BERT keyword similarity."""
    user_embedding = bert_model.encode(user_text, convert_to_tensor=True)

    # Compute max similarity scores with risk keyword embeddings
    high_risk_score = torch.max(util.cos_sim(user_embedding, high_risk_embeds)).item()
    moderate_risk_score = torch.max(util.cos_sim(user_embedding, moderate_risk_embeds)).item()
    low_risk_score = torch.max(util.cos_sim(user_embedding, low_risk_embeds)).item()

    # Determine highest similarity
    scores = {
        "High-Risk": high_risk_score,
        "Moderate Concern": moderate_risk_score,
        "Low Concern": low_risk_score
    }


    return max(scores, key=scores.get)
# Function to predict risk level
# def predict_text(text):
#     sequences = tokenizer.texts_to_sequences([text])
#     padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=128)
#     prediction = model.predict(padded_sequences)
#     predicted_class = np.argmax(prediction, axis=-1)[0]
#     return risk_labels[predicted_class]

# Function to extract location
def extract_location(text):
    if isinstance(text, str):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":  # "GPE" = Geo-Political Entity (City, State, Country)
                return ent.text
    return None

def geocode_location(place):
    try:
        location = geolocator.geocode(place)
        return (location.latitude, location.longitude) if location else None
    except:
        return None

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard/")
def dashboard():
    return render_template("dashboard.html")

def store_df_in_sql(df):
    with app.app_context():
        for _, row in df.iterrows():
            existing_post = RedditPost.query.filter_by(post_id=row["Post ID"]).first()

            if not existing_post:  # Insert only if post_id and timestamp do not exist
                post = RedditPost(
                    subreddit=row["Subreddit"],
                    post_id=row["Post ID"],
                    timestamp=pd.to_datetime(row["Timestamp"], unit='s'),
                    title=row["Title"],
                    content=row["Content"],
                    upvotes=row["Upvotes"],
                    comments=row["Comments"],
                    url=row["URL"],
                    sentiment=row["Sentiment"],
                    risk_level=row["Risk Level"]
                )
                db.session.add(post)

        db.session.commit()

def fetch_data():
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
    query = "SELECT * FROM reddit_posts"
    d = pd.read_sql(query, engine)
    print(d.info())
    return d

def create_heatmap():

    df_clone =fetch_data()
    print(df_clone.info())

    df_clone['location'] = df_clone['content'].apply(lambda x: extract_location(str(x)))
    df_clone = df_clone.dropna()
    df_clone['location'] = df_clone['location'].map(country_mapping)
    
    valid_countries = list(set(country_mapping.values()))
    df_clone.loc[~df_clone['location'].isin(valid_countries), 'location'] = 'Unknown'
    
    df_clone['location'] = df_clone['location'].str.title()
    print(df_clone.info())
    df_clone['Coordinates'] = df_clone['location'].apply(lambda x: geocode_location(x))
    df_clone = df_clone.dropna()
    
    if df_clone.empty:
        print("No valid coordinates found.")
        return None
    print("Coordinations completed")
    print(df_clone.info())
    df_clone['Latitude'] = df_clone['Coordinates'].apply(lambda x: x[0])
    df_clone['Longitude'] = df_clone['Coordinates'].apply(lambda x: x[1])
    
    try:
        m = folium.Map(location=[df_clone['Latitude'].mean(), df_clone['Longitude'].mean()], zoom_start=5)
        heat_data = list(zip(df_clone['Latitude'], df_clone['Longitude']))
        HeatMap(heat_data).add_to(m)
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None

    heatmap_path = "static/heat_map.html"
    m.save(heatmap_path)
    return heatmap_path

@app.route('/analyze/', methods=['GET'])
def analyze_posts():
    global df
    df = extract_reddit_posts()
    df["Cleaned Content"] = df["Content"].apply(clean_text)
    df["Cleaned Title"] = df["Title"].apply(clean_text)

    df = df.dropna()
    df["Sentiment"] = df["Cleaned Content"].apply(sentiment_scores)
    df["Risk Level"] = df["Cleaned Content"].apply(classify_risk_level)
    store_df_in_sql(df)

    return jsonify({"message": "Data stored in SQL successfully!", "rows": len(df)})

@app.route("/heatmap/")
def heatmap():
    heatmap_file = create_heatmap()
    if heatmap_file:
        return render_template("heatmap.html", heatmap_file=heatmap_file)
    else:
        return "<h1>Error: Missing Latitude and Longitude data.</h1>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
