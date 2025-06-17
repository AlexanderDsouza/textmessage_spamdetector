# Spam Message Classifier with Explanation (Flask + TF-IDF + Naive Bayes)

A simple spam detection web app built with Python, scikit-learn, and Flask.  
It uses a Multinomial Naive Bayes model trained on the classic SMS Spam Collection dataset, with TF-IDF text features.  
The app also provides an explanation highlighting the words most contributing to the spam or ham classification.

---

## Features

- Trains a spam classifier on SMS text data.
- Cleans and preprocesses messages (lowercase, remove punctuation).
- Uses TF-IDF vectorization with English stopword removal.
- Multinomial Naive Bayes classification model.
- Flask backend serving:
  - Prediction endpoint (`/predict`) returning spam/ham label and confidence.
  - Word-level explanation with contribution scores.
- Responsive front-end UI:
  - Input box for messages.
  - Dark mode toggle.
  - Displays prediction, confidence, and top contributing words with colored highlights.

---

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/spam-classifier-flask.git
cd spam-classifier-flask

