import os
import time

import pandas as pd
import requests
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages (like success/error)

# Initialize the Groq API key from the environment variable
API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama3-8b-8192"  # Default model for sentiment analysis

# Define allowed file types
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Function to check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process the uploaded file
def process_file(file):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    if 'Review' not in df.columns:
        return None, "Missing 'Review' column in file."
    
    reviews = df['Review'].dropna().tolist()
    return reviews, None

# Function to classify the sentiment as positive, negative, or neutral
def classify_sentiment(sentiment_content):
    sentiment_content_lower = sentiment_content.lower()
    
    if 'positive' in sentiment_content_lower:
        return 'positive'
    elif 'negative' in sentiment_content_lower:
        return 'negative'
    else:
        return 'neutral'

# Function to analyze sentiment using Groq API
def analyze_sentiment(review):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the sentiment of this review: '{review}'"
            }
        ]
    }

    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            response.raise_for_status()  # Raises an error for non-200 responses
            response_data = response.json()

            # Print API response for debugging
            print(f"API Response: {response_data}")

            # Extract the content of the sentiment analysis
            choices = response_data.get('choices')
            if choices and len(choices) > 0:
                sentiment_content = choices[0].get('message', {}).get('content', None)
                if sentiment_content:
                    return {"sentiment": sentiment_content}
                else:
                    return {"error": "Sentiment content not found in the response."}
            else:
                return {"error": "No choices found in the API response."}

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit error
                print(f"Rate limit reached. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return {"error": f"HTTP error: {str(e)}"}

        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    return {"error": "Failed after multiple retries due to rate limiting."}


# Web route for the file upload form
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            reviews, error = process_file(file)
            if error:
                flash(error)
                return redirect(request.url)
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            results = []
            
            for review in reviews:
                sentiment_result = analyze_sentiment(review)

                # Handle the case where sentiment is not present due to an error
                if 'sentiment' in sentiment_result:
                    sentiment = classify_sentiment(sentiment_result['sentiment'])  # Classify the sentiment
                else:
                    sentiment = 'unknown'  # Fallback in case of API failure or error
                    flash(f"Error analyzing sentiment for review: {sentiment_result.get('error', 'Unknown error')}")

                # Count sentiment classification
                if sentiment == 'positive':
                    positive_count += 1
                elif sentiment == 'negative':
                    negative_count += 1
                else:
                    neutral_count += 1

                results.append({
                    "review": review,
                    "sentiment": sentiment  # Save classified sentiment here
                })

            total_reviews = len(reviews)
            positive_score = round(positive_count / total_reviews, 2)
            negative_score = round(negative_count / total_reviews, 2)
            neutral_score = round(neutral_count / total_reviews, 2)
            
            sentiment_summary = {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            }

            return render_template('results.html', results=results, summary=sentiment_summary)
        
        else:
            flash('Allowed file types are CSV, XLSX')
            return redirect(request.url)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

