# import os
# import torch
# import whisper
# import joblib
# import numpy as np
# from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename
# from sklearn.feature_extraction.text import CountVectorizer

# app = Flask(__name__)

# # Load trained model and vectorizer
# clf = joblib.load("burmese_decision_tree_model.joblib")
# cv = joblib.load("burmese_vectorizer.joblib")




# # Flask Routes
# # ✅ Route for Text Detection
# @app.route("/", methods=["GET", "POST"])
# def text_detect():
#     if request.method == "POST":
#         user_text = request.form.get("text")
        
#         if not user_text:
#             return jsonify({"error": "No text provided"})
        
#         # Transform text for prediction
#         test_data = cv.transform([user_text]).toarray()
#         prediction = clf.predict(test_data)[0]
        
#         return jsonify({
#             "input_text": user_text,
#             "prediction": prediction
#         })
    
#     return render_template("burmese.html")

# if __name__ == "__main__":
#     app.run(debug=True)


import os
import torch
import whisper
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Whisper AI model (same model for both languages)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# Load trained classifiers for both languages
clf_en = joblib.load("decision_tree_model.joblib")  # English classifier
cv_en = joblib.load("count_vectorizer.joblib")

clf_mm = joblib.load("burmese_decision_tree_model.joblib")  # Burmese language classifier
cv_mm = joblib.load("burmese_vectorizer.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/test")
def about():
    return render_template("test.html")

# ✅ Route for Text Detection (Multi-Language)
@app.route("/text_detect", methods=["GET", "POST"])
def text_detect():
    if request.method == "POST":
        user_text = request.form.get("text")
        language = request.form.get("language")  # User selects 'english' or 'other'

        if not user_text or not language:
            return jsonify({"error": "Text or language not provided"})

        if language == "english":
            test_data = cv_en.transform([user_text]).toarray()
            prediction = clf_en.predict(test_data)[0]
        else:
            test_data = cv_mm.transform([user_text]).toarray()
            prediction = clf_mm.predict(test_data)[0]

        return jsonify({
            "input_text": user_text,
            "language": language,
            "prediction": prediction
        })

    return render_template("text_detect.html")

# ✅ Route for Audio Detection (Multi-Language)
@app.route("/detect", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files["audio"]
        language = request.form.get("language")  # User selects 'english' or 'other'

        if file.filename == "" or not language:
            return jsonify({"error": "No selected file or language not specified"})

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Transcribe audio using Whisper
        result = whisper_model.transcribe(file_path)
        transcribed_text = result["text"]

        # Choose classifier based on language
        if language == "english":
            test_data = cv_en.transform([transcribed_text]).toarray()
            prediction = clf_en.predict(test_data)[0]
        else:
            test_data = cv_mm.transform([transcribed_text]).toarray()
            prediction = clf_mm.predict(test_data)[0]

        return jsonify({
            "transcribed_text": transcribed_text,
            "language": language,
            "prediction": prediction
        })

    return render_template("detect.html")

if __name__ == "__main__":
    app.run(debug=True)
