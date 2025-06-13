from flask import Flask, request, jsonify, render_template
import pickle

# Load model and vectorizer
with open('saved_models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('saved_models/tfidf.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

import os

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'Missing "text" field'}), 400

    input_text = [data['text']]
    vect_text = tfidf.transform(input_text).toarray()
    pred = model.predict(vect_text)[0]
    result = 'Spam' if pred == 1 else 'Not Spam'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
