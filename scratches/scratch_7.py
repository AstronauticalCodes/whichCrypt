import joblib
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

model = joblib.load('E4-Optimized-AES-CAST.joblib')

sample_text = ''.join("OkBVWCn/WMUocA==".split(" "))
predict_encryption_algorithm(model, sample_text, label_encoder)
