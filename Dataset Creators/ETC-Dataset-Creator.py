from Crypto.Cipher import AES, ARC2, ARC4
from Crypto.Cipher import Blowfish, CAST, ChaCha20
from Crypto.Cipher import ChaCha20_Poly1305, DES, DES3
from Crypto.Cipher import PKCS1_OAEP, PKCS1_v1_5, Salsa20
from Crypto.Random import get_random_bytes
from Crypto.Cipher import ARC4
from Crypto.Hash import SHA
from Crypto import Random
from Crypto.PublicKey import RSA
import base64
from csv import writer
from random import shuffle
import numpy as np
import re
import rsa
from pycipher import Caesar, Playfair, ColTrans, Vigenere
from EncryptionDataset.rareCiphersWhichCrypt import playfairGenerator, affineGenerator, railfenceGenerator, blowfishGenertor


# with open('bigAF.txt', encoding='utf-8') as file:
with open('Bigger-wC.txt', encoding='utf-8') as file:
    data = file.read().split('\n')
    data = [x for x in data if len(x) != 0]
    data = data[:len(data)]

with open('E12-TD-wC-AES-DES.csv', 'w', encoding='utf-8', newline='') as csvFile:
    writer = writer(csvFile)

    encryptRows = []
    writer.writerow(['Encrypted-Text', 'Encryption'])
    i = 0
    # while i < len(data)
    for line in data:

            # caesar = Caesar(key=3)
            # ciphertext = caesar.encipher(line)
            # if ciphertext != "":
            #     encryptRows.append([ciphertext, "Caesar"])
            #
            # vigenere = Vigenere(key="keyword")
            # ciphertext = vigenere.encipher(line)
            # if ciphertext != "":
            #     encryptRows.append([ciphertext, "Vignere"])
            #
            # # encryptRows.append([playfairGenerator(line), "Playfair"])
            # encryptRows.append([railfenceGenerator(line), "Railfence"])


        key = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_EAX)
        nonce = cipher.nonce
        encryptCypherText, tag = cipher.encrypt_and_digest(line.encode(encoding='utf-8'))
        encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        encryptRows.append([encoded_encryptCypherText, 'AES'])

        key = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_EAX)
        nonce = cipher.nonce
        encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        encryptRows.append([encoded_encryptCypherText, "DES"])

        # key = get_random_bytes(24)
        # cipher = DES3.new(key, DES3.MODE_EAX)
        # nonce = cipher.nonce
        # encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # encryptRows.append([encoded_encryptCypherText, 'DES3'])
        # # #
        # key = get_random_bytes(16)
        # cipher = CAST.new(key, CAST.MODE_EAX)
        # nonce = cipher.nonce
        # encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # encryptRows.append([encoded_encryptCypherText, "CAST"])


        # encryptRows.append([blowfishGenertor(line), "Blowfish"])

        # key = get_random_bytes(32)
        # cipher = Salsa20.new(key=key)
        # nonce = cipher.nonce
        # encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # encryptRows.append([encoded_encryptCypherText, 'Salsa20'])
        #     #
        # key = get_random_bytes(32)
        # cipher = ChaCha20.new(key=key)
        # nonce = cipher.nonce
        # encryptCypherText = cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
        # if encoded_encryptCypherText != "":
        #     encryptRows.append([encoded_encryptCypherText, 'ChaCha20'])
        # #
        # key = b'YourSecretKey'
        # nonce = Random.new().read(16)
        # tempkey = SHA.new(key + nonce).digest()
        # cipher = ARC4.new(tempkey)
        # ciphertext = nonce + cipher.encrypt(line.encode(encoding='utf-8'))
        # encoded_encryptCypherText = base64.b64encode(ciphertext).decode('utf-8')
        # if encoded_encryptCypherText != "":
        #     encryptRows.append([encoded_encryptCypherText, 'ARC4'])

        # except Exception:
        #     print('haf')
        #     continue

            # i += 1
            # print(i)
    shuffle(encryptRows)

    writer.writerows(encryptRows)
