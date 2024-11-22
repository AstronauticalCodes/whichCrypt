import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from scipy.fft import fft
from pywt import wavedec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('E3-TD-wC-AES-DES.csv')

# Assuming the CSV has columns 'text' and 'encryption'
encryptedText = data['Encrypted Text'].values
encryption = data['Encryption'].values

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(encryption)
labels_categorical = to_categorical(labels_encoded)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
features_tfidf = tfidf_vectorizer.fit_transform(encryptedText).toarray()

# Feature extraction using n-grams
ngram_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
features_ngram = ngram_vectorizer.fit_transform(encryptedText).toarray()

# Feature extraction using character-level features
def extract_char_features(texts):
    features = []
    for text in texts:
        feature = [ord(char) for char in text]
        features.append(feature[:100])  # Truncate or pad to fixed length
    return np.array(features)

features_char = extract_char_features(encryptedText)

# Feature extraction using DCT
def extract_dct_features(texts):
    features = []
    for text in texts:
        feature = dct([ord(char) for char in text], norm='ortho')
        features.append(feature[:100])  # Truncate or pad to fixed length
    return np.array(features)

features_dct = extract_dct_features(encryptedText)

# Feature extraction using DFT
def extract_dft_features(texts):
    features = []
    for text in texts:
        feature = np.abs(fft([ord(char) for char in text]))
        features.append(feature[:100])  # Truncate or pad to fixed length
    return np.array(features)

features_dft = extract_dft_features(encryptedText)

# Feature extraction using Wavelet Transform
def extract_wavelet_features(texts):
    features = []
    for text in texts:
        coeffs = wavedec([ord(char) for char in text], 'db1', level=2)
        feature = np.hstack(coeffs)
        features.append(feature[:100])  # Truncate or pad to fixed length
    return np.array(features)

features_wavelet = extract_wavelet_features(encryptedText)

# Combine features
features = np.hstack((features_tfidf, features_ngram, features_char, features_dct, features_dft, features_wavelet))

# Dimensionality reduction using PCA
pca = PCA(n_components=500)
features_pca = pca.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_pca, labels_categorical, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(1024, input_shape=(features_pca.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('encryption_classifier_model.h5')

# Function to manually check encryption type
def check_encryption_type(text):
    feature_tfidf = tfidf_vectorizer.transform([text]).toarray()
    feature_ngram = ngram_vectorizer.transform([text]).toarray()
    feature_char = extract_char_features([text])
    feature_dct = extract_dct_features([text])
    feature_dft = extract_dft_features([text])
    feature_wavelet = extract_wavelet_features([text])
    feature_combined = np.hstack((feature_tfidf, feature_ngram, feature_char, feature_dct, feature_dft, feature_wavelet))
    feature_pca = pca.transform(feature_combined)
    prediction = model.predict(feature_pca)
    encryption_type = label_encoder.inverse_transform([np.argmax(prediction)])
    return encryption_type[0]

# Example usage
encrypted_text = "your_encrypted_text_here"
print(f'The encryption type is: {check_encryption_type(encrypted_text)}')
