import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Load dataset
data = pd.read_csv('/home/user/Desktop/whichCrypt/P0+-TD-wC.csv')

# Verify column names
print(data.columns.tolist())

# Strip leading/trailing spaces
data.columns = data.columns.str.strip()

# Ensure column names are strings
data.columns = data.columns.astype(str)

# Check if 'Cipher Text' column exists
if 'Cipher Text' not in data.columns:
    raise KeyError("Column 'Cipher Text' not found in the dataset")

# Print a sample of the data
print(data.head())

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
data['features'] = data.get('Cipher Text').apply(extract_features)
X = pd.DataFrame(data['features'].tolist())
y = data['Cipher']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape data for RNN
X_train_rnn = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the RNN model
model = Sequential([
    SimpleRNN(50, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test_rnn), axis=-1)
print(f'Accuracy: {100*(accuracy_score(y_test, y_pred))}')

# Prediction function
def predict_algorithm(hash_str):
    features = extract_features(hash_str)
    features_df = pd.DataFrame([features])
    features_rnn = np.array(features_df).reshape((features_df.shape[0], features_df.shape[1], 1))
    probabilities = 100*(model.predict(features_rnn)[0])
    algorithms = label_encoder.classes_
    return dict(zip(algorithms, probabilities))

# Example usage
hash_input = 'bef0a85a10723404d0402c9b7c2bc7ad27017b7d702cf1f831dce80f4410a0ef'
predictions = predict_algorithm(hash_input)
print(predictions)
