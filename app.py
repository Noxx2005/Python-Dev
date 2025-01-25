# main.py
import re  
import string 
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
""" from sklearn.naive_bayes import MultinomialNB """
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


app = Flask(__name__, static_folder="static")
CORS(app)

stop_words = set(stopwords.words('english'))
stemmer = LancasterStemmer()

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic\S+', '', text)
    text = re.sub(r'[^a-zA-Z+]', ' ', text)
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stop_words and len(i) > 2])
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load the dataset
df = pd.read_csv("data/Phishing_Email.csv", encoding="latin-1", usecols=["Email Text", "Email Type"])
df = df.dropna(axis=0)
df.columns = ["Email Text", "Email Type"]

# Clean the text data
df["CleanMail"] = df["Email Text"].apply(cleaning_data)

# Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["CleanMail"])
y = df["Email Type"].map({'Safe Email': 0, 'Phishing Email': 1})
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sms_input = data['smsInput']
    
    # Preprocess the input data
    cleaned_input = cleaning_data(sms_input)
    
    # Transform input using the same vectorizer used for training
    input_transformed = vectorizer.transform([cleaned_input])
    
    # Make predictions and get confidence scores
    prediction = model.predict(input_transformed)[0]
    probabilities = model.predict_proba(input_transformed)[0]  # Confidence scores
    
    # Confidence score for the predicted class
    confidence_score = max(probabilities) * 100  # Convert to percentage
    
    # Return result with confidence score
    if prediction == 0:
        result = "The mail is safe to open"
    else:
        result = "Phishing email!! Be careful"
    
    return jsonify({'result': result, 'confidence': f"{confidence_score:.2f}%"})


if __name__ == '__main__':
    app.run(debug=True)
