import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load dataset
data = pd.read_csv('/home/user/Desktop/whichCrypt/P0+-TD-wC.csv')

# Verify column names
data.columns = data.columns.str.strip()
data.columns = data.columns.astype(str)

# Check if 'Cipher Text' column exists
if 'Cipher Text' not in data.columns:
    raise KeyError("Column 'Cipher Text' not found in the dataset")

# Feature extraction function
def extract_features(hash_str):
    features = {
        'length': len(hash_str),
        'digit_count': sum(c.isdigit() for c in hash_str),
        'alpha_count': sum(c.isalpha() for c in hash_str),
        # Add more features as needed
    }
    return features

# Prepare the dataset
data['features'] = data['Cipher Text'].apply(extract_features)
X = pd.DataFrame(data['features'].tolist())
y = data['Cipher']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Train the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=1))  # Adjusted pool size to avoid negative dimension
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Convert features to numpy array for CNN
X_train_cnn = np.array(X_train)
X_test_cnn = np.array(X_test)

# Reshape for CNN input
X_train_cnn = X_train_cnn.reshape((X_train_cnn.shape[0], X_train_cnn.shape[1], 1))
X_test_cnn = X_test_cnn.reshape((X_test_cnn.shape[0], X_test_cnn.shape[1], 1))

cnn_model = create_cnn_model((X_train_cnn.shape[1], 1))
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Combine predictions
rf_predictions = rf_model.predict_proba(X_test)
cnn_predictions = cnn_model.predict(X_test_cnn)

# Average the probabilities
combined_predictions = (rf_predictions + cnn_predictions) / 2
final_predictions = np.argmax(combined_predictions, axis=1)

# Evaluate the combined model
accuracy = accuracy_score(y_test, final_predictions)
print(f'Ensemble Model Accuracy: {accuracy * 100:.2f}%')
