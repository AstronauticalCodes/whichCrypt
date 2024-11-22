import csv
import numpy as np
from hashlib import sha256, sha224, sha384, sha512, sha3_224, sha3_256, sha3_384, sha3_512, sha1
from hashlib import md5
from hashlib import blake2b, blake2s
from hashlib import shake_256, shake_128
import hashlib

with open("Big-wC.txt", encoding='utf-8') as file:
    data = file.read().split('\n')
    data = data[:len(data)//1]

with open("HFC0-TD-wC-.csv", 'w', newline='', encoding='utf-8') as csvFile:
    writer = csv.writer(csvFile)
    features = ["Hash", "Entropy", "Length", "Unigram_Dist", "Bigram"]
    label = ["Hash-Function"]

    hashRows = []
    def calculate_entropy(text):
        """Calculate entropy to measure the randomness in the encrypted text."""
        prob_dist = np.bincount(np.frombuffer(text.encode('utf-8'), dtype=np.uint8)) / len(text)
        return -np.sum([p * np.log2(p) for p in prob_dist if p > 0])

    def char_distribution(text, vectorizer):
        """Calculate the frequency distribution of characters in the encrypted text."""
        return vectorizer.transform([text]).toarray().flatten()

    for line in data:
        #SHA Family
        hashRows.append([sha256(line.encode(encoding='utf-8')).hexdigest()])
