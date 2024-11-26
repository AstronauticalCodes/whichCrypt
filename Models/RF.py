import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = pd.read_csv('/home/user/Desktop/whichCrypt/B0-TD-wC.csv')

# Verify column names
print(data.columns.tolist())

# Strip leading/trailing spaces
data.columns = data.columns.str.strip()

# Ensure column names are strings
data.columns = data.columns.astype(str)

# Check if 'hash' column exists
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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {100*(accuracy_score(y_test, y_pred))}')

# Prediction function
def predict_algorithm(hash_str):
    features = extract_features(hash_str)
    features_df = pd.DataFrame([features])
    probabilities = 100*(model.predict_proba(features_df)[0])
    algorithms = model.classes_
    return dict(zip(algorithms, probabilities))

# Example usage
# hash_input = ''.join('4D 0F D0 D2 A0 09 F5 10 E0 8A 30 06 4D 53 A4 1F 63 4A 90 29'.split(" "))
hash_input = 'bef0a85a10723404d0402c9b7c2bc7ad27017b7d702cf1f831dce80f4410a0ef'

predictions = predict_algorithm(hash_input)
print(predictions)