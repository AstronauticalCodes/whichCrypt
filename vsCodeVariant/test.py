import model

# Step 5: Custom testing function
def predict_algorithm(hash_str):
    features = np.array(hash_to_features(hash_str)).reshape(1, -1)
    return model.predict(features)[0]

# Example usage
custom_hash = '4D0FD0D2A009F510E08A30064D53A41F634A9029'  # Example MD5 hash
print(f'The algorithm used is: {predict_algorithm(custom_hash)}')