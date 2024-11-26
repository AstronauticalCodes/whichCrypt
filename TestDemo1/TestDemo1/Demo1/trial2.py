import json

# Data to be written
data = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "is_student": False,
    "courses": ["Math", "Science", "History"]
}

# Writing to sample.json
with open("sample.json", "w") as json_file:
    json.dump(data, json_file, indent=4)
