from Crypto.Cipher import AES, ARC2, ARC4
from Crypto.Cipher import Blowfish, CAST, ChaCha20
from Crypto.Cipher import ChaCha20_Poly1305, DES3, DES
from Crypto.Cipher import PKCS1_OAEP, PKCS1_v1_5, Salsa20
from Crypto.Random import get_random_bytes
import hashlib
import base64
import csv
from random import shuffle

with open("NotThatBigAF.txt", encoding='utf-8') as file:
    data = file.read().split('\n')
    data = [x for x in data if len(x) != 0]
    data = data[:len(data)//3000]

with open("EH1-TD-wC-AESDES-SHASHA3.csv", 'w', newline='', encoding='utf-8') as dataset:
    writer = csv.writer(dataset)

    features = ['Cipher Text', 'Cipher Type']

    writer.writerow(features)

    cipherRows = []

    for line in data:
        key = get_random_bytes(32)
        cipher = AES.new(key, AES.MODE_EAX)
        nonce = cipher.nonce
        encryptCypherText, tag = cipher.encrypt_and_digest(line.encode(encoding='utf-8'))
        encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        cipherRows.append([encoded_encryptCypherText, 'Encryption'])

    # for line in data:
        key = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_EAX)
        nonce = cipher.nonce
        encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        cipherRows.append([encoded_encryptCypherText, 'DES'])

        # key = get_random_bytes(32)
        # cipher = Blowfish.new(key, Blowfish.MODE_EAX)
        # nonce = cipher.nonce
        # encryptCypherText, tag = cipher.encrypt_and_digest(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # HashRows.append([encoded_encryptCypherText, 'Encryption'])

        # key = get_random_bytes(32)
        # cipher = ChaCha20.new(key=key)
        # nonce = cipher.nonce
        # encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # HashRows.append([encoded_encryptCypherText, 'Encryption'])

        # key = get_random_bytes(32)
        # cipher = ARC4.new(key)
        # encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # HashRows.append([encoded_encryptCypherText, 'Encryption'])

        hash = hashlib.md5(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'Hash'])

        hash = hashlib.sha1(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'Hash'])

        # hash = hashlib.blake2b(line.encode(encoding='utf-8')).hexdigest()
        # HashRows.append([hash, 'Hash'])

        hash = hashlib.sha256(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'sha256'])

        hash = hashlib.sha3_256(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'Hash'])

        hash = hashlib.sha384(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'sha384'])

        hash = hashlib.sha3_384(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'Hash'])

        hash = hashlib.sha224(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'sha224'])

        hash = hashlib.sha3_224(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'sha3_224'])

        hash = hashlib.sha512(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'sha512'])

        hash = hashlib.sha3_512(line.encode(encoding='utf-8')).hexdigest()
        cipherRows.append([hash, 'sha3_512'])

    shuffle(cipherRows)

    writer.writerows(cipherRows)
