from Crypto.Cipher import AES, ARC2, ARC4
from Crypto.Cipher import Blowfish, CAST, ChaCha20
from Crypto.Cipher import ChaCha20_Poly1305, DES, DES3
from Crypto.Cipher import PKCS1_OAEP, PKCS1_v1_5, Salsa20
from Crypto.Random import get_random_bytes
from Crypto.PublicKey import RSA
import base64
from csv import writer
from random import shuffle
import numpy as np
import re
import rsa
from pycipher import Caesar, Playfair, ColTrans, Vigenere
from rareCiphersWhichCrypt import playfairGenerator, affineGenerator, railfenceGenerator

# with open('bigAF.txt', encoding='utf-8') as file:
with open('Big-wC.txt', encoding='utf-8') as file:
    data = file.read().split('\n')
    data = [x for x in data if len(x) != 0]
    data = data[:len(data)]
    # print(len(data))


encryptRows = []

def calculate_entropy(text):
    """Calculate entropy to measure the randomness in the encrypted text."""
    prob_dist = np.bincount(np.frombuffer(text.encode('utf-8'), dtype=np.uint8)) / len(text)
    return -np.sum([p * np.log2(p) for p in prob_dist if p > 0])

def text_length(text):
    """Get the length of the encrypted text."""
    return len(text)

def char_distribution(text, vectorizer):
    """Calculate the frequency distribution of characters in the encrypted text."""
    return vectorizer.transform([text]).toarray().flatten()


def special_char_count(text):
    """Calculate the frequency of special characters in the text."""
    special_chars = re.findall(r'[!@#$%^&*(),.?":{}|<>]', text)
    return len(special_chars)

def appender(encrypted_text, encryption):
    global encryptRows
    length = text_length(encrypted_text)
    entropy = calculate_entropy(encrypted_text)
    special_chars = special_char_count(encrypted_text)
    encryptRows.append([encrypted_text, length, entropy, special_chars, encryption])


with open('E8-TD-wC-.csv', 'w', encoding='utf-8', newline='') as csvFile:
    writer = writer(csvFile)

    writer.writerow(['Encrypted Text', "Length", "Entropy", "SpecialChars", 'Encryption'])
    i = 0
    while i < len(data):
        line = data[i]
        # RSA
        public_key, private_key = rsa.newkeys(512)
        try:
            encrypted_message = rsa.encrypt(line.encode(), public_key).decode('utf-16')
            appender(encrypted_message, "RSA")
        except Exception:
            i-= 1
            continue

        caesar = Caesar(key=3)
        ciphertext = caesar.encipher(line)
        appender(ciphertext, "Caesar")

        vigenere = Vigenere(key="keyword")
        ciphertext = vigenere.encipher(line)
        appender(ciphertext, "Vignere")

        appender(playfairGenerator(line), "Playfair")

        key = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_EAX)
        nonce = cipher.nonce
        encryptCypherText, tag = cipher.encrypt_and_digest("abhinaasfasdfasdfsav".encode(encoding='utf-8'))
        encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        appender(encoded_encryptCypherText, "AES")

        appender(affineGenerator(line), "Affine")

        appender(railfenceGenerator(line), "Railfence")

        appender(ColTrans("line").encipher(line), "ColTrans")


        # for line in data:
        # key = get_random_bytes(8)
        # cipher = DES.new(key, DES.MODE_EAX)
        # nonce = cipher.nonce
        # encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # appender(encoded_encryptCypherText, "DES")

        # for line in data:
        #     key = get_random_bytes(32)
        #     cipher = ARC4.new(key)
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     encryptRows.append([encoded_encryptCypherText, 'ARC4'])
        #
        # for line in data:
        #     key = get_random_bytes(16)
        #     cipher = Blowfish.new(key, Blowfish.MODE_EAX)
        #     nonce = cipher.nonce
        #     encryptCypherText, tag = cipher.encrypt_and_digest(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     appender(encoded_encryptCypherText, "Blowfish")
            # encryptRows.append([encoded_encryptCypherText, 'Blowfish'])
        # #
        # for line in data:
        #     key = get_random_bytes(16)
        #     cipher = CAST.new(key, CAST.MODE_EAX)
        #     nonce = cipher.nonce
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     appender(encoded_encryptCypherText, "CAST")
        # #
        # for line in data:
        #     key = get_random_bytes(32)
        #     cipher = ChaCha20.new(key=key)
        #     nonce = cipher.nonce
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     appender(encoded_encryptCypherText, "ChaCha20")
        #
        # for line in data:
        #     key = get_random_bytes(32)
        #     cipher = ChaCha20_Poly1305.new(key=key)
        #     nonce = cipher.nonce
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     encryptRows.append([encoded_encryptCypherText, 'ChaCha20_Poly1305'])

        # for line in data:
        #     key = get_random_bytes(24)
        #     cipher = DES3.new(key, DES3.MODE_EAX)
        #     nonce = cipher.nonce
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     encryptRows.append([encoded_encryptCypherText, 'DES3'])

        # for line in data:
        #     key = RSA.generate(2048)
        #     publicKey = key.public_key().exportKey()
        #     rsaKey = RSA.importKey(publicKey)
        #     cipher = PKCS1_OAEP.new(rsaKey)
        #     # nonce = cipher.nonce
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     encryptRows.append([encoded_encryptCypherText, 'PKCS1_OAEP'])

        # for line in data:
        #     key = RSA.generate(2048)
        #     publicKey = key.public_key().exportKey()
        #     rsaKey = RSA.importKey(publicKey)
        #     cipher = PKCS1_v1_5.new(rsaKey)
        #     # nonce = cipher.nonce
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     encryptRows.append([encoded_encryptCypherText, 'PKCS1_v1_5'])


        # for line in data:
        #     key = get_random_bytes(32)
        #     cipher = Salsa20.new(key=key)
        #     nonce = cipher.nonce
        #     encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        #     encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        #     encryptRows.append([encoded_encryptCypherText, 'Salsa20'])

    shuffle(encryptRows)

    writer.writerows(encryptRows)
