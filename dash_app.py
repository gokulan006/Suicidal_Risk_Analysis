import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from sqlalchemy.exc import SQLAlchemyError
from flask_apscheduler import APScheduler
import requests
# Function to Create Dashboard
def create_dashboard(flask_app, db):
    dash_app = dash.Dash(
        server=flask_app,
        name="Dashboard",
        url_base_pathname="/dashboard/"
    )

    # Import Model (Better than defining inside function)
    from app import RedditPost  

    # Fetch Data from DB
    def fetch_data():
        try:
            with flask_app.app_context():
                query = db.session.query(RedditPost.sentiment, RedditPost.risk_level).all()
                df = pd.DataFrame(query, columns=["Sentiment", "Risk Level"])
                return df
        except SQLAlchemyError as e:
            print(f"Database error: {e}")
            return pd.DataFrame(columns=["Sentiment", "Risk Level"])  # Return empty DataFrame on error

    # Dashboard Layout
    dash_app.layout = html.Div([
        html.H1("Risk Level & Sentiment Distribution"),
        
        dcc.Graph(id="risk-level-pie"),
        dcc.Graph(id="sentiment-bar"),

        dcc.Interval(
            id="interval-component",
            interval=5000,  # Refresh every 5 seconds
            n_intervals=0
        ),
    ])

    # Update Graphs
    @dash_app.callback(
        [dash.Output("risk-level-pie", "figure"),
         dash.Output("sentiment-bar", "figure")],
        [dash.Input("interval-component", "n_intervals")]
    )
    def update_graphs(n):
        df = fetch_data()

        if df.empty:
            return px.pie(title="No Data Available"), px.bar(title="No Data Available")

        # Pie Chart for Risk Level
        risk_fig = px.pie(df, names="Risk Level", title="Risk Level Distribution")

        # Bar Chart for Sentiment
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        sentiment_fig = px.bar(sentiment_counts, x="Sentiment", y="Count", title="Sentiment Distribution")

        return risk_fig, sentiment_fig

    return dash_app

scheduler = APScheduler()

def schedule_tasks():
    print("Running scheduled tasks...")

    try:
        requests.get("http://127.0.0.1:5000/analyze")  # Extract & Analyze
        print("✔ /analyze executed successfully")

        requests.get("http://127.0.0.1:5000/dashboard")  # Refresh Dashboard
        print("✔ /dashboard executed successfully")

        requests.get("http://127.0.0.1:5000/heatmap")  # Refresh Heatmap
        print("✔ /heatmap executed successfully")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")

# Register the scheduler in Flask App
def init_scheduler(app):
    scheduler.init_app(app)
    scheduler.start()
    scheduler.add_job(id="Scheduled Task", func=schedule_tasks, trigger="interval", minutes=15)
