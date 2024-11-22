# import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# import tensorflow as tf
# from sklearn.externals import joblib
import pickle

data = pd.read_csv('EH0-TD-wC.csv')

print(data.columns.tolist())

data.columns = data.columns.str.strip()

data.columns = data.columns.astype(str)

if 'Cipher Text' not in data.columns:
    raise KeyError("Column 'hash' not found in the dataset")

print(data.head())

def extract_features(hash_str):
    features = {
        'length': len(hash_str),
        'digit_count': sum(c.isdigit() for c in hash_str),
        'alpha_count': sum(c.isalpha() for c in hash_str)
    }
    return features

data['features'] = data.get('Cipher Text').apply(extract_features)
X = pd.DataFrame(data['features'].tolist())
y = data['Cipher Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# model.save("TFModel.h5")
# joblib.dump(model, "Trial0.pkl")
pickle.dump(model, open('T-encryptHashModel.pkl', 'wb'))

def predict_algorithm(hash_str):
    features = extract_features(hash_str)
    features_df = pd.DataFrame([features])
    probabilities = model.predict_proba(features_df)[0]
    algorithms = model.classes_
    return dict(zip(algorithms, probabilities))

# Example usage
# hash_input = ''.join('4D 0F D0 D2 A0 09 F5 10 E0 8A 30 06 4D 53 A4 1F 63 4A 90 29'.split(" "))
hash_input = ''.join('996499120a303d5c22fde5920207dbeda000b7177b2da922cb89270a4a0a19566801c25ce60b91855ec962c7c97d036187'.split(" "))

predictions = predict_algorithm(hash_input)
print(predictions)
