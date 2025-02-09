import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import torch
import whisper
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename


# Load dataset
file_path = r"./updated_burmese_hatespeech_and_normalspeech_dataset.csv"
df = pd.read_csv(file_path)

# Burmese Text Cleaning Function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^\u1000-\u109F ]', '', text)  # Keep only Burmese characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["Text"] = df["Text"].apply(clean)

# Features and labels
x = np.array(df["Text"])
y = np.array(df["Label"])

# Text vectorization
cv = CountVectorizer(analyzer='char', ngram_range=(1, 3))
x = cv.fit_transform(df["Text"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(clf, 'burmese_decision_tree_model.joblib')
joblib.dump(cv, "burmese_vectorizer.joblib")

print("\nModel and vectorizer saved successfully.")

# Model Evaluation
print("\nTraining Accuracy:", clf.score(X_train, y_train))
print("Testing Accuracy:", clf.score(X_test, y_test))

print("\nClassification Report:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))
