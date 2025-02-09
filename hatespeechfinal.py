
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import whisper
import joblib
import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))
import pickle
import torch

df = pd.read_csv(r"./hatespeech_dataset.csv")
print(df.head())

df['labels'] = df['class'].map({0:"Hate Speech Detected", 1:"Offensive language detected", 2:"No hate and offensive speech"})
print(df.head())

df = df[['tweet', 'labels']]
df.head()

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '',text)
    text = re.sub(r'https?://\S+|www\.\S+', '',text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)
    return text

df["tweet"] = df["tweet"].apply(clean)
print(df.head())

df = df.dropna(subset=["labels"])
print(df.isnull().sum())  # Verify no NaN values

x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
x  = cv.fit_transform(x)
joblib.dump(cv, 'count_vectorizer.joblib')

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state= 42)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

print("Training Accuracy:", clf.score(X_train, y_train))
print("Testing Accuracy:", clf.score(X_test, y_test))

import joblib

# Save the trained model to a file
joblib.dump(clf, 'decision_tree_model.joblib')

# Load the trained model from the file
clf_loaded = joblib.load('decision_tree_model.joblib')

# ✅ Load Whisper AI model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# ✅ Load trained model
clf = joblib.load("decision_tree_model.joblib")  # Ensure this file exists

# ✅ Set the audio file path
audio_file_path = "intro.wav"

# Step 1: Transcribe audio using Whisper AI
result = whisper_model.transcribe(audio_file_path)
transcribed_text = result["text"]
print(f"🎤 Transcribed Text: {transcribed_text}")

test_data = transcribed_text
df = cv.transform([test_data]).toarray()
print (clf.predict(df))

