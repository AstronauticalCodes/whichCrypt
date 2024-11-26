import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('/home/user/Desktop/whichCrypt/EH0-TD-wC.csv')
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
y = data['Cipher Type']
# Feature selection and labels
# X = data['']  # Replace with actual feature names
# y = data['label']  # 'encryption' or 'hashing'

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {100*(accuracy_score(y_test, y_pred))}')

joblib.dump(model,'EH0.joblib')

# Manual input prediction
def classify_algorithm(features):
    return model.predict([features])

# Example usage
# manual_input = [value1, value2, value3]  # Replace with actual input values
# print(f'The input is classified as: {classify_algorithm(manual_input)}')