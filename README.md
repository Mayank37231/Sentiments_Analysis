# Flask Sentiment Analysis API

This is a Flask-based web application that allows users to upload a CSV or XLSX file containing customer reviews. The app integrates with the Groq API to perform sentiment analysis on the reviews and displays both a summary and detailed results.

## Features
- Upload CSV or XLSX files with customer reviews.
- Analyze the sentiment (positive, neutral, or negative) of each review using the Groq API.
- Display a summary of positive, negative, and neutral reviews.
- User-friendly interface styled with Bootstrap for a clean and responsive design.

## Technologies Used
- **Flask**: Web framework for building the API and web app.
- **Pandas**: For processing CSV and XLSX files.
- **Groq API**: To perform sentiment analysis.
- **Bootstrap**: For styling the web interface.

## Setup Instructions

### Prerequisites
- Python 3.8+
- A Groq API Key (you can get one [here](https://console.groq.com/))
- Git

### Local Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/rishabhjohri/flask-sentiment-analysis.git
   cd flask-sentiment-analysis
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   //On Windows use 
    venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Set up environment variables:
   Create a .env file in the root of your project and add your Groq API key:
    ```bash
    GROQ_API_KEY=your-groq-api-key
5. Run the application:
  ```bash
  python app.py




