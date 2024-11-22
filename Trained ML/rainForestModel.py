import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Collect and preprocess the dataset
# For simplicity, let's assume you have a CSV file with two columns: 'hash' and 'algorithm'
data = pd.read_csv('B0-TD-wC.csv')

# Convert hashes to numerical features (e.g., ASCII values)

MAX_LENGTH = 64

def hash_to_features(hash_str):
    features = [ord(char) for char in hash_str]
    if len(features) > MAX_LENGTH:
        return features[:MAX_LENGTH]
    else:
        return features + [0] * (MAX_LENGTH - len(features))

data['features'] = data['Cipher Text'].apply(hash_to_features)
X = np.array(data['features'].tolist())
y = data['Cipher']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
