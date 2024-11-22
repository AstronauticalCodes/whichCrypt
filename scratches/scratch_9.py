import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.fftpack import fft, dct
import pywt

# Load Data
data = pd.read_csv('E3-TD-wC-AES-DES.csv')
X = data['Encrypted Text']
y = data['Encryption']

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Character-level Features using n-grams
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_char = vectorizer.fit_transform(X)

# TF-IDF Features
tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_tfidf = tfidf_vectorizer.fit_transform(X)


# Function to pad sequences to a fixed length
def pad_sequence(seq, max_len):
    if len(seq) > max_len:
        return np.array(seq[:max_len])  # Truncate if too long
    else:
        return np.pad(seq, (0, max_len - len(seq)), 'constant')  # Pad if too short


# DFT Features
max_len = max(len(text) for text in X)
X_dft = np.array([pad_sequence(np.abs(fft(list(map(ord, text)))), max_len) for text in X])

# DCT Features
X_dct = np.array([pad_sequence(dct(list(map(ord, text))), max_len) for text in X])


# Wavelet Transform Features
def wavelet_features(text):
    coeffs = pywt.wavedec(list(map(ord, text)), 'db1', level=2)
    return np.concatenate(coeffs)


X_wavelet = np.array([pad_sequence(wavelet_features(text), max_len) for text in X])

# Combine all features
X_combined = np.hstack((X_char.toarray(), X_tfidf.toarray(), X_dft, X_dct, X_wavelet))

# Dimensionality Reduction using PCA
pca = PCA(n_components=200)
X_reduced = pca.fit_transform(X_combined)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=42)

# Models
# KNN
knn = KNeighborsClassifier(n_neighbors=5)

# RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# SVM
svm = SVC(probability=True, random_state=42)


# CNN Model for feature extraction
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# RNN Model for feature extraction
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Prepare data for CNN and RNN
# CNN and RNN models require sequences of a fixed length
X_combined_padded = pad_sequences(X_combined, maxlen=max_len)

# Train CNN and RNN Models
cnn_model = create_cnn_model((X_combined_padded.shape[1], 1))
rnn_model = create_rnn_model((X_combined_padded.shape[1], 1))

# Reshape data for CNN and RNN
X_combined_padded_cnn = X_combined_padded[..., np.newaxis]
X_combined_padded_rnn = X_combined_padded[..., np.newaxis]

cnn_model.fit(X_combined_padded_cnn, y_encoded, epochs=10, batch_size=32, validation_split=0.2)
rnn_model.fit(X_combined_padded_rnn, y_encoded, epochs=10, batch_size=32, validation_split=0.2)


# Custom Voting Classifier
class CustomVotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        for name, estimator in self.estimators:
            if hasattr(estimator, 'fit'):
                estimator.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(label_encoder.classes_)))
        for name, estimator in self.estimators:
            if hasattr(estimator, 'predict_proba'):
                predictions += estimator.predict_proba(X)
            elif hasattr(estimator, 'predict'):
                predictions += np.expand_dims(estimator.predict(X), axis=1)
        return np.argmax(predictions, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# Combine all models
custom_voting_clf = CustomVotingClassifier([
    ('knn', knn),
    ('rf', rf),
    ('svm', svm),
    ('cnn', cnn_model),
    ('rnn', rnn_model)
])

# Fit custom voting classifier
custom_voting_clf.fit(X_train, y_train)

# Evaluate the custom voting classifier
accuracy = custom_voting_clf.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')


# Function to manually check the encryption algorithm for a given text
def check_encryption_algorithm(encrypted_text):
    # Extract features
    char_features = vectorizer.transform([encrypted_text]).toarray()
    tfidf_features = tfidf_vectorizer.transform([encrypted_text]).toarray()
    dft_features = pad_sequence(np.abs(fft(list(map(ord, encrypted_text)))), max_len)
    dct_features = pad_sequence(dct(list(map(ord, encrypted_text))), max_len)
    wavelet_feature = pad_sequence(wavelet_features(encrypted_text), max_len)

    # Combine all features
    combined_features = np.hstack((char_features, tfidf_features, dft_features, dct_features, wavelet_feature))

    # Apply PCA transformation to match the feature set used during training
    reduced_features = pca.transform(combined_features.reshape(1, -1))

    # Prepare data for CNN and RNN
    combined_padded = pad_sequences(reduced_features, maxlen=max_len)

    # Reshape data for CNN and RNN
    combined_padded_cnn = combined_padded[..., np.newaxis]
    combined_padded_rnn = combined_padded[..., np.newaxis]

    # Get predictions
    cnn_proba = cnn_model.predict(combined_padded_cnn)
    rnn_proba = rnn_model.predict(combined_padded_rnn)

    # Combine predictions from different models
    combined_proba = np.mean([cnn_proba, rnn_proba], axis=0)

    # Predict using the custom voting classifier
    prediction = custom_voting_clf.predict(reduced_features)

    return label_encoder.inverse_transform(prediction)[0]


# Option to manually check encryption algorithm
encrypted_text = input("Enter encrypted text to check the encryption algorithm: ").strip().replace(" ", "")
algorithm = check_encryption_algorithm(encrypted_text)
print(f'The encryption algorithm used is: {algorithm}')
