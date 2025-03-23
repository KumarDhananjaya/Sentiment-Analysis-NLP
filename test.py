import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Ensure required NLTK data is downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Load the cleaned dataset
df = pd.read_csv("data/test.csv", header=None, names=["label", "title", "text"])

# Handle missing values
df = df.fillna("")

# Combine title and text
df["full_text"] = df["title"] + " " + df["text"]

# Text Preprocessing Function
def clean_text(text):
    if not isinstance(text, str):  # Ensure text is a string
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Apply cleaning to text
df["cleaned_text"] = df["full_text"].apply(clean_text)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["label"], test_size=0.2, random_state=42)

# Create a text classification pipeline (TF-IDF + Naive Bayes)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),  
    ("classifier", MultinomialNB())  
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict sentiment for all reviews
df["predicted_sentiment"] = pipeline.predict(df["cleaned_text"])

# Save the results to a new file
df.to_csv("data/test_sentiment_ml.csv", index=False)
print("Sentiment analysis completed! Results saved in 'data/test_sentiment_ml.csv'.")

# Visualization of Sentiment Distribution
# Enhanced Visualization of Sentiment Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=df["predicted_sentiment"], palette="magma", edgecolor="black", alpha=0.85)

# Add labels and title with better styling
plt.xlabel("Predicted Sentiment", fontsize=14, fontweight="bold")
plt.ylabel("Count", fontsize=14, fontweight="bold")
plt.title("Sentiment Analysis Results", fontsize=16, fontweight="bold")

# Show grid for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display value counts on bars
for p in plt.gca().patches:
    plt.gca().annotate(
        f"{int(p.get_height())}", 
        (p.get_x() + p.get_width() / 2, p.get_height()), 
        ha="center", 
        va="bottom", 
        fontsize=12, 
        fontweight="bold",
        color="black"
    )

plt.show()
