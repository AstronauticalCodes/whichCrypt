import joblib
import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft
from scipy.fftpack import dct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def preprocess_text(text):
    # Convert text to lower case
    text = text.lower()
    # Remove non-alphanumeric characters
    text = ''.join(c for c in text if c.isalnum())
    return text


def extract_features(text):
    features = []

    # Character-level features
    count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
    char_features = count_vectorizer.fit_transform([text]).toarray()[0]
    features.extend(char_features)

    # DFT features
    dft_features = np.abs(fft(np.array(list(text.encode('utf-8')))))
    features.extend(dft_features)

    # DCT features
    dct_features = np.abs(dct(np.array(list(text.encode('utf-8'))), norm='ortho'))
    features.extend(dct_features)

    # Wavelet Transform features
    text_int = np.array(list(text.encode('utf-8')), dtype=np.float32)
    if len(text_int) > 1:
        level = min(4, int(np.log2(len(text_int))))
        wavelet_coeffs = pywt.wavedec(text_int, 'haar', level=level)
        for coeff in wavelet_coeffs:
            # Normalize the length of each coefficient array
            if len(coeff) < 512:
                coeff = np.pad(coeff, (0, 512 - len(coeff)), 'constant')
            elif len(coeff) > 512:
                coeff = coeff[:512]
            features.extend(coeff)

    # Ensure features length consistency
    features = np.array(features)
    if len(features) < 2048:
        features = np.pad(features, (0, 2048 - len(features)), 'constant')
    elif len(features) > 2048:
        features = features[:2048]

    return features


def train_model(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(probability=True),  # Enable probability for SVM
        'KNN': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
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
        # Get probability predictions
        probabilities = model.predict_proba([features])[0]

        # Map the probabilities to the corresponding labels
        labels = label_encoder.classes_
        label_probabilities = {label: prob for label, prob in zip(labels, probabilities)}

        # Sort by probability in descending order
        sorted_probabilities = sorted(label_probabilities.items(), key=lambda x: x[1], reverse=True)

        # Display the label and its corresponding probability
        print(f"\n{name} Prediction Probabilities:")
        for label, prob in sorted_probabilities:
            print(f"{label}: {prob * 100:.2f}%")


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
    models = train_model(X_train, y_train)

    # Evaluate models
    evaluate_model(models, X_test, y_test, label_encoder)
    joblib.dump(models, "E4-Trial-AES-CAST.joblib")
    # Predict on a new encrypted text
    sample_text = ''.join("JIxBj9YZAeXJsA==".split(" "))
    predict_encryption_algorithm(models, sample_text, label_encoder)


if _name_ == "_main_":
    main()
