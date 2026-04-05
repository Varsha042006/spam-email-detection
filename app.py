from flask import Flask, render_template_string, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and train model once

data = pd.read_csv("spam_ham_dataset.csv")

if 'label' in data.columns and 'text' in data.columns:
    data = data[['label', 'text']]
    data.columns = ['label', 'message']

if data['label'].dtype == 'object':
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'\\d+', '', text)
    text = re.sub(r'\\s+', ' ', text)
    return text


data['message'] = data['message'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(data['message'])
y = data['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Simple HTML
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Spam Detector</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        input { width: 300px; padding: 10px; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <h2>Spam Email Classifier</h2>
    <form method="post">
        <input type="text" name="message" placeholder="Enter message" required>
        <br><br>
        <button type="submit">Check</button>
    </form>
    <h3>{{ result }}</h3>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        msg = request.form['message']
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        result = "Spam" if pred == 1 else "Not Spam"
    return render_template_string(html, result=result)

if __name__ == '__main__':
    app.run(debug=True)
