import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK datasets
nltk.download('stopwords')

# Load the dataset and limit to 500 rows
df = pd.read_csv('IMDB.csv').head(500)

# Convert sentiment to binary (1 for positive, 0 for negative)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Text preprocessing
stop_words = list(set(stopwords.words('english')))
count_vect = CountVectorizer(stop_words=stop_words)
tfidf_transformer = TfidfTransformer()

# Fit and transform the training data
X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Transform the test data
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Train classifiers
nb_classifier = MultinomialNB().fit(X_train_tfidf, y_train)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_tfidf, y_train)
svm_classifier = SVC(kernel='linear', C=1, random_state=42).fit(X_train_tfidf, y_train)

def get_user_input_and_predict():
    # Accept user input
    user_review = input("Enter a movie review: ")
    
    # Preprocess the user input in the same way as the training data
    user_review_processed = count_vect.transform([user_review])
    user_review_tfidf = tfidf_transformer.transform(user_review_processed)
    
    # Predict sentiment using each model
    nb_prediction = nb_classifier.predict(user_review_tfidf)[0]
    rf_prediction = rf_classifier.predict(user_review_tfidf)[0]
    svm_prediction = svm_classifier.predict(user_review_tfidf)[0]
    
    print(f"\nNaive Bayes Prediction: {'Positive' if nb_prediction == 1 else 'Negative'}")
    print(f"Random Forest Prediction: {'Positive' if rf_prediction == 1 else 'Negative'}")
    print(f"SVM Prediction: {'Positive' if svm_prediction == 1 else 'Negative'}")

def evaluate_models_with_cross_validation():
    # Define the models
    models = [
        ("Naive Bayes", MultinomialNB()),
        ("Random Forest", RandomForestClassifier(n_estimators=100)),
        ("SVM", SVC(kernel='linear', C=1))
    ]
    
    # Prepare the full dataset for cross-validation
    X_full = count_vect.transform(df['review'])
    X_full_tfidf = tfidf_transformer.transform(X_full)
    y_full = df['sentiment']
    
    # Perform stratified k-fold cross-validation for each model
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, X_full_tfidf, y_full, cv=kfold, scoring='accuracy')
        print(f"{name} Accuracy: {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")

# Evaluate models using cross-validation
evaluate_models_with_cross_validation()

# Get user input and predict sentiment
get_user_input_and_predict()
