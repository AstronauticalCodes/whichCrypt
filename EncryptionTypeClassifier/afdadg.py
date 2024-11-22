
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load dataset
data = pd.read_csv('B0-TD-wC.csv')

data.columns = data.columns.str.strip()

# print(data.head())

def extract_features(hash_str):
    features = {
        'length': len(hash_str),
        'digit_count': sum(c.isdigit() for c in hash_str),
        'alpha_count': sum(c.isalpha() for c in hash_str)
    }
    return features

data['features'] = data.get('Cipher Text').apply(extract_features)
X = pd.DataFrame(data['features'].tolist())
y = data['Cipher']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# model.save('srijanAccuracy.h5')


# Prediction function
def predict_algorithm(hash_str):
    features = extract_features(hash_str)
    features_df = pd.DataFrame([features])
    probabilities = model.predict_proba(features_df)[0]
    algorithms = model.classes_
    return dict(zip(algorithms, probabilities))


# dump(model, open('srijanAccuracy.sav', 'wb'))
# mlflow.sklearn.log_model(model, 'model')

joblib.dump(model, 'RFCe.joblib')

# Example usage
# hash_input = ''.join('4D 0F D0 D2 A0 09 F5 10 E0 8A 30 06 4D 53 A4 1F 63 4A 90 29'.split(" "))
#
# predictions = predict_algorithm(hash_input)
# print(predictions)
