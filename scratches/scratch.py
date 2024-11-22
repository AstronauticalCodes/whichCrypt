import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = pd.read_csv('B0-TD-wC.csv')

# Verify column names
print(data.columns.tolist())

# Strip leading/trailing spaces
data.columns = data.columns.str.strip()

# Ensure column names are strings
data.columns = data.columns.astype(str)

# Check if 'hash' column exists
if 'Cipher Text' not in data.columns:
    raise KeyError("Column 'hash' not found in the dataset")

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
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
