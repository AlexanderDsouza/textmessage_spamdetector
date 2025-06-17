import numpy as np
import pandas as pd
import joblib
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df = pd.read_csv('spam.csv', encoding='latin1')
df = df[['v1', 'v2']]

# Rename columns for clarity
df.columns = ['label', 'message']

# Map labels to numeric values for classification
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

df['clean_message'] = df['message'].apply(preprocess_text)


# TF-IDF Vectorizer (removes English stop words)
vectorizer = TfidfVectorizer(stop_words='english')

# Convert text data to TF-IDF features
X = vectorizer.fit_transform(df['clean_message'])

# Labels
y = df['label_num']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f'Training samples: {X_train.shape[0]}')
print(f'Test samples: {X_test.shape[0]}')


model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Save the model
joblib.dump(model, 'spam_classifier_model.joblib')
# Save the vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')



# Load the saved model and vectorizer
model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def predict_message(message):
    # Preprocess the message
    clean = preprocess_text(message)
    
    # Transform using the TF-IDF vectorizer
    vectorized = vectorizer.transform([clean])
    
    # Predict using the trained model
    prediction = model.predict(vectorized)[0]
    
    # Convert numeric prediction back to label
    label = 'spam' if prediction == 1 else 'ham'
    
    return label

# Try it with your own messages
messages = [
    "Congratulations! You've won a free cruise to the Bahamas!",
    "Hey, are we still on for lunch today?",
    "URGENT! Your account has been suspended. Click here to verify.",
    "why god",
    "hi"
]

for msg in messages:
    print(f'Message: {msg}')
    print(f'Prediction: {predict_message(msg)}\n')
