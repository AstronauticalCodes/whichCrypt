import joblib
import json
import pandas as pd
from hashlib import sha256

prevUserHash = ''
while True:
    with open('CSPY-Model.json') as cspyModelJson:
        data = cspyModelJson.read()
        modelJson = json.loads(data)
        userModelName = modelJson['Model'] + '.joblib'

    # with open(f'Models\\{userModelName}', 'rb') as modelFile:
    #     model = joblib.load(modelFile)
    model = joblib.load(f"Models\\{userModelName}")

    def extract_features(hash_str):
        features = {
            'length': len(hash_str),
            'digit_count': sum(c.isdigit() for c in hash_str),
            'alpha_count': sum(c.isalpha() for c in hash_str)
        }
        return features

    def predict_algorithm(hash_str):
        features = extract_features(hash_str)
        features_df = pd.DataFrame([features])
        probabilities = model.predict_proba(features_df)[0]
        algorithms = model.classes_
        return dict(zip(algorithms, probabilities))

    with open('CSPY-Hash.json') as cspyHashJson:
        data = cspyHashJson.read()
        hashJson = json.loads(data)
        userHash = hashJson['Hash']

    if userHash != prevUserHash:
        predictions = predict_algorithm(userHash)
        print(predictions)
        predList = []
        for x in predictions:
            if predictions[x] > 0:
                predList.append(f'{x} -> {predictions[x]}')

        with open('PYCS-Pred.json', 'w') as pycsPredFile:
            json.dump({"Predictions": predList}, pycsPredFile, indent=4)

        prevUserHash = userHash

    with open('CSPY-Exit.json') as cspyExitJson:
        data = cspyExitJson.read()
        exitJson = json.loads(data)
        exitCode = int(exitJson['Exit'])


    # if exitCode:
    #     exit()
