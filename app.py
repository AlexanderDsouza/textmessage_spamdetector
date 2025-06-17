from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')


def get_word_contributions(message, vectorizer, model):
    tokens = message.lower().split()
    tfidf_vector = vectorizer.transform([message])
    feature_names = vectorizer.get_feature_names_out()

    # Log probability difference between spam and ham
    log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    
    word_scores = []
    for word in tokens:
        if word in feature_names:
            idx = vectorizer.vocabulary_.get(word)
            if idx is not None:
                tfidf_value = tfidf_vector[0, idx]
                score = tfidf_value * log_prob_diff[idx]
                word_scores.append((word, round(score, 4)))
    
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_scores


@app.route('/', methods=['GET'])
def index():
    # Serves your HTML front-end
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    vect = vectorizer.transform([message])
    prediction = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0][prediction]
    word_scores = get_word_contributions(message, vectorizer, model)

    return jsonify({
        'message': message,
        'prediction': int(prediction),
        'label': 'spam' if prediction == 1 else 'ham',
        'confidence': round(float(proba) * 100, 2),
        'explanation': word_scores  # List of (word, score)
    })


if __name__ == '__main__':
    app.run(debug=True)
