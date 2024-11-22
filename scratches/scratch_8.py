import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.fftpack import fft, dct
import pywt

# Load Data
data = pd.read_csv('E4-mini-TD-wC-AES-CAST.csv')
X = data['Encrypted Text']
y = data['Encryption']

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Character-level Features using n-grams
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_char = vectorizer.fit_transform(X)

# Function to pad sequences to a fixed length (ensures same length for all sequences)
def pad_sequence(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)), 'constant')

# DFT Features (Discrete Fourier Transform)
max_len = max(len(text) for text in X)  # Determine max length in the dataset
X_dft = np.array([pad_sequence(np.abs(fft(list(map(ord, text)))), max_len) for text in X])

# DCT Features (Discrete Cosine Transform)
X_dct = np.array([pad_sequence(dct(list(map(ord, text))), max_len) for text in X])

# Wavelet Transform Features
def wavelet_features(text):
    # Apply Discrete Wavelet Transform and concatenate the coefficients
    coeffs = pywt.wavedec(list(map(ord, text)), 'db1', level=2)
    return np.concatenate(coeffs)

# Extract Wavelet Features for all texts
X_wavelet = np.array([pad_sequence(wavelet_features(text), max_len) for text in X])

# Combine all features (Character-level, DFT, DCT, Wavelet)
X_combined = np.hstack((X_char.toarray(), X_dft, X_dct, X_wavelet))

# Dimensionality Reduction using PCA (reduce to 100 components)
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_combined)

# Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=42)

# Train Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # RandomForest
svm = SVC(probability=True, random_state=42)  # Support Vector Machine
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)  # Multi-layer Perceptron (Neural Network)

# Voting Classifier: Combines RandomForest, SVM, and MLP (using soft voting)
voting_clf = VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)], voting='soft')
voting_clf.fit(X_train, y_train)

# Evaluate the model on the test set and print the accuracy
accuracy = voting_clf.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Function to manually check the encryption algorithm for a given text
def check_encryption_algorithm(encrypted_text):
    # Extract character-level n-gram features
    char_features = vectorizer.transform([encrypted_text]).toarray()  # Already 2D

    # Extract DFT, DCT, and Wavelet features
    dft_features = pad_sequence(np.abs(fft(list(map(ord, encrypted_text)))), max_len).reshape(1, -1)
    dct_features = pad_sequence(dct(list(map(ord, encrypted_text))), max_len).reshape(1, -1)
    wavelet_feature = pad_sequence(wavelet_features(encrypted_text), max_len).reshape(1, -1)

    # Combine all the features
    combined_features = np.hstack((char_features, dft_features, dct_features, wavelet_feature))

    # Apply PCA transformation to match the feature set used during training
    reduced_features = pca.transform(combined_features)

    # Predict the encryption algorithm
    prediction = voting_clf.predict(reduced_features)

    # Return the predicted encryption algorithm (decoded from the label)
    return label_encoder.inverse_transform(prediction)[0]

# Example Usage: Predict encryption algorithm for a given encrypted text
encrypted_text = ''.join('''zfmsZpt49kpnkvSSXPw4zZJ/81dgZq2Pwx7BWKCf5EuMjo7ibM0tTvEvf9E7A7g3jAC7b4w/Z+jtwvjmfBjsQ3DtJpwIeaWxcm0/yATYzq0U/rE/6EPonKk5/XA='''.split(" "))
algorithm = check_encryption_algorithm(encrypted_text)
print(f'The encryption algorithm used is: {algorithm}')
