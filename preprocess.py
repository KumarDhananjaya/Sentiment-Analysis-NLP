import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK datasets
nltk.download("stopwords")
nltk.download("punkt")

# Load dataset
df = pd.read_csv("data/test.csv")

# Check and rename columns if needed
expected_columns = ["rating", "title", "text"]
if list(df.columns)[:3] != expected_columns:
    df.columns = expected_columns

# Drop missing values
df.dropna(subset=["text"], inplace=True)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing
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

# Encode sentiment labels
le = LabelEncoder()
df["sentiment_encoded"] = le.fit_transform(df["sentiment"])

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment_encoded"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict sentiment of new text
def predict_sentiment(review):
    review = preprocess_text(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    return le.inverse_transform(prediction)[0]

# Test predictions
print("\nExample Predictions:")
print(predict_sentiment("This product is amazing! I love it."))
print(predict_sentiment("This is the worst experience ever."))
