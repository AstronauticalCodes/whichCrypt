import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft
from scipy.fftpack import dct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_text(text):
    # Convert text to lower case
    text = text.lower()
    # Remove non-alphanumeric characters
    text = ''.join(c for c in text if c.isalnum())
    return text


def extract_features(text):
    features = []

    # Character-level features (Unigram and Bigram only to reduce dimensionality)
    count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2), max_features=100)
    char_features = count_vectorizer.fit_transform([text]).toarray()[0]
    features.extend(char_features)

    # DFT features (limit to first 100 coefficients)
    dft_features = np.abs(fft(np.array(list(text.encode('utf-8')))))[:100]
    features.extend(dft_features)

    # DCT features (limit to first 100 coefficients)
    dct_features = np.abs(dct(np.array(list(text.encode('utf-8'))), norm='ortho'))[:100]
    features.extend(dct_features)

    # Wavelet Transform features (reduce decomposition level to 2 and limit coefficients)
    text_int = np.array(list(text.encode('utf-8')), dtype=np.float32)
    if len(text_int) > 1:
        wavelet_coeffs = pywt.wavedec(text_int, 'haar', level=2)
        for coeff in wavelet_coeffs:
            coeff = coeff[:50]  # Limit to first 50 coefficients
            features.extend(coeff)

    # Ensure features length consistency (pad or trim to 500)
    features = np.array(features)
    if len(features) < 500:
        features = np.pad(features, (0, 500 - len(features)), 'constant')
    elif len(features) > 500:
        features = features[:500]

    return features


def train_model(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10),  # Reduced depth and estimators
        'LinearSVC': LinearSVC(max_iter=1000),  # Linear SVC instead of RBF kernel SVM
        'KNN': KNeighborsClassifier(n_neighbors=3),  # Limit neighbors
        'ExtraTrees': ExtraTreesClassifier(n_estimators=50),  # Faster ensemble method
        'LogisticRegression': LogisticRegression(max_iter=500)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained.")

    return models


def evaluate_model(models, X_test, y_test, label_encoder):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Predictions: {y_pred_labels[:5]}")  # Display a few predictions as labels


def predict_encryption_algorithm(models, text, label_encoder):
    features = extract_features(preprocess_text(text))

    for name, model in models.items():
        # Handle case where some models do not support `predict_proba` (e.g., LinearSVC)
        try:
            probabilities = model.predict_proba([features])[0]
            labels = label_encoder.classes_
            label_probabilities = {label: prob for label, prob in zip(labels, probabilities)}
            sorted_probabilities = sorted(label_probabilities.items(), key=lambda x: x[1], reverse=True)

            print(f"\n{name} Prediction Probabilities:")
            for label, prob in sorted_probabilities:
                print(f"{label}: {prob * 100:.2f}%")
        except AttributeError:
            # Fall back to direct prediction for models without `predict_proba`
            prediction = model.predict([features])[0]
            label = label_encoder.inverse_transform([prediction])[0]
            print(f"\n{name} Prediction: {label}")


def main():
    # Load and preprocess dataset
    df = pd.read_csv('E4-TD-wC-AES-CAST.csv')
    df['Encrypted Text'] = df['Encrypted Text'].apply(preprocess_text)

    # Extract features
    X = np.array([extract_features(text) for text in df['Encrypted Text']])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Encryption'])

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    # models = train_model(X_train, y_train)
    models = joblib.load("E4-Optimized-AES-CAST.joblib")

    # Evaluate models
    # evaluate_model(models, X_test, y_test, label_encoder)
    # joblib.dump(models, "E4-Optimized-AES-CAST.joblib")

    # Predict on a new encrypted text
    sample_text = ''.join("LpUdGyxu2r5O1vX+Jf2uXzXaS946/Q==".split(" "))
    predict_encryption_algorithm(models, sample_text, label_encoder)


if __name__ == "__main__":
    main()
