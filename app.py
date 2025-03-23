from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import subprocess


# Initialize Flask
app = Flask(__name__)

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Global variables for model and vectorizer
model = None
vectorizer = None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

def train_model():
    global model, vectorizer

    # Load dataset
    df = pd.read_csv("data/test.csv")

    # Preprocess text
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Convert ratings to sentiment labels
    def get_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating <= 2:
            return "negative"
        else:
            return "neutral"

    df["sentiment"] = df["rating"].apply(get_sentiment)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["sentiment"], test_size=0.2, random_state=42)

    # Train model
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vectors, y_train)

    # Evaluate model
    predictions = model.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, predictions)
    
    return {"accuracy": accuracy, "predictions": predictions.tolist(), "actual": y_test.tolist()}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    results = train_model()
    return jsonify({"accuracy": results["accuracy"]})

def test():
    # subprocess.run(["python", "test.py"]÷ß)  # Run the test script
    
    # Read the saved CSV
    df = pd.read_csv("data/test_sentiment_ml.csv")
    
    # Count occurrences of each sentiment
    sentiment_counts = df["Predicted Sentiment"].value_counts().to_dict()

    # Read classification report JSON
    with open("data/classification_report.json", "r") as f:
        classification_report = jsonify.load(f)
    
    return jsonify({
        "accuracy": classification_report["accuracy"],
        "sentiment_counts": sentiment_counts,
        "classification_report": classification_report
    })

if __name__ == "__main__":
    app.run(debug=True)
