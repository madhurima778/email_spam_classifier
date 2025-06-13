# Email Spam Classifier

A machine learning project to classify emails as **Spam** or **Not Spam** using NLP techniques and various classifiers. The project includes a Flask web app for easy prediction.

---

## Features

- Data cleaning and preprocessing (tokenization, stopword removal, stemming)
- Exploratory Data Analysis (EDA) and visualization
- Feature extraction using TF-IDF and CountVectorizer
- Handling class imbalance with SMOTE
- Model training with multiple algorithms (Naive Bayes, Logistic Regression, SVM, Random Forest, etc.)
- Model evaluation and comparison
- Flask web app for real-time predictions

---

## Project Structure

```
email_spam_classifier/
│
├── app.py                  # Flask web application
├── email_spam_classifier.ipynb  # Jupyter notebook with all ML code
├── saved_models/
│   ├── model.pkl           # Trained ML model
│   └── tfidf.pkl           # Trained TF-IDF vectorizer
├── Data/
│   └── spam.csv            # Raw dataset
├── index.html              # Web app frontend (Flask template)
└── README.md               # Project documentation
```

---

## Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/email_spam_classifier.git
    cd email_spam_classifier
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is missing, install manually:*
    ```bash
    pip install flask scikit-learn pandas numpy matplotlib seaborn nltk wordcloud imbalanced-learn xgboost
    ```

3. **Download NLTK data**
    In Python:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

4. **Run the Flask app**
    ```bash
    python app.py
    ```
    The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Usage

- Open the web app in your browser.
- Enter/paste an email message.
- Click **Predict** to see if it is spam or not.

---

## Model Training

All data processing, feature engineering, model training, and evaluation steps are in `email_spam_classifier.ipynb`.  
You can retrain the model or experiment with different algorithms and parameters there.

---

## Notes

- The model uses SMOTE to balance the dataset. For text data, also consider class weighting or simple oversampling.
- The Flask app loads the trained model and vectorizer from the `saved_models` directory.
- Make sure `index.html` is in the same directory as `app.py` or set the correct `template_folder` in Flask.

---
