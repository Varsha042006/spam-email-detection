# 1. Import Libraries

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# 📌 2. Load Dataset

data = pd.read_csv("spam_ham_dataset.csv")

# Fix columns
if 'label' in data.columns and 'text' in data.columns:
    data = data[['label', 'text']]
    data.columns = ['label', 'message']

# Convert labels
if data['label'].dtype == 'object':
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# 📌 3. Clean Text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['message'] = data['message'].apply(clean_text)


# 📌 4. Convert Text → Numbers

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))

X = vectorizer.fit_transform(data['message'])
y = data['label']


# 📌 5. Train Model

model = LogisticRegression(max_iter=1000)
model.fit(X, y)


# 📌 6. User Input Only

while True:
    msg = input("\nEnter a message (or type 'exit'): ")

    if msg.lower() == 'exit':
        print("Exiting...")
        break

    cleaned_msg = clean_text(msg)
    msg_vector = vectorizer.transform([cleaned_msg])

    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("🚨 Spam")
    else:
        print("✅ Not Spam")