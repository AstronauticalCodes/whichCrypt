from csv import writer
import rsa
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP


with open('Bigger-wC.txt', encoding='utf-8') as file:
    data = file.read().split('\n')
    data = [x for x in data if len(x) != 0]
    data = data[:len(data)]

with open('E10-TD-wC-AES-SCRATCH.csv', 'w', encoding='utf-8', newline='') as csvFile:
    writer = writer(csvFile)

    encryptRows = []
    writer.writerow(['Encrypted-Text', 'Encryption'])
    i = 0
    while i < len(data):
        line = data[i]
        # RSA
        public_key, private_key = rsa.newkeys(512)
        # try:
            #     encrypted_message = rsa.encrypt(line.encode(), public_key).decode('utf-16')
            #     encryptRows.append([encrypted_message, "Block"])
            #     print(encrypted_message)
            # Save keys to files
            # with open('private.pem', 'wb') as f:
            #     f.write(private_key)
            # with open('public.pem', 'wb') as f:
            #     f.write(public_key)
            #
            # # Load public key for encryption
            # with open('public.pem', 'rb') as f:
            #     public_key = RSA.import_key(f.read())
            #
            # # Encrypt message
            # message = line.encode('utf-8')
            # cipher = PKCS1_OAEP.new(public_key)
            # encrypted_message = cipher.encrypt(message)
            # print("Encrypted message:", encrypted_message)
        # public_key, private_key = rsa.newkeys(512)
        try:
            encrypted_message = rsa.encrypt(line.encode(), public_key).decode('utf-16')
            print(encrypted_message)
            # encryptRows.append([encrypted_message, "Block"])

        except Exception:
            print("helo")
            i -=1
            continue
        #
        # except Exception:
        #     # print("helo")
        #     # i -= 1
        #     continue
#         # Generate RSA keys
#         key = RSA.generate(2048)
#         private_key = key.export_key()
#         public_key = key.publickey().export_key()
#         #
#         # Save keys to files
#         with open('private.pem', 'wb') as f:
#             f.write(private_key)
#         with open('public.pem', 'wb') as f:
#             f.write(public_key)
#
#         # Load public key for encryption
#         with open('public.pem', 'rb') as f:
#             public_key = RSA.import_key(f.read())
#
#         # Encrypt message
#         message = line.encode('utf-8')
#         cipher = PKCS1_OAEP.new(public_key)
#         encrypted_message = cipher.encrypt(message)
#         print("Encrypted message:", encrypted_message)
#
        i +=1
# import base64
#
# from Crypto.Cipher import Salsa20
# from Crypto.Random import get_random_bytes
#
# key = get_random_bytes(32)
# cipher = Salsa20.new(key=key)
# nonce = cipher.nonce
# encryptCypherText = cipher.encrypt("asasdfline".encode(encoding='utf-8'))
# encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
# # encryptRows.append([encoded_encryptCypherText, 'Salsa20'])
# print(encoded_encryptCypherText)

# from twofish import Twofish
#
# # Define your key (must be 16, 24, or 32 bytes long)
# key = b'This is a key123'
#
# # Create a Twofish instance
# T = Twofish(key)
#
# # Define the plaintext (must be a multiple of 16 bytes)
# plaintext = b'YELLOWSUBMARINES'
#
# # Encrypt the plaintext
# ciphertext = T.encrypt(plaintext)
# #
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.backends import default_backend
#
# key = b'This is a key123'
# cipher = Cipher(algorithms.Twofish(key), modes.ECB(), backend=default_backend())
#
# encryptor = cipher.encryptor()
# plaintext = b'YELLOWSUBMARINES'
# ciphertext = encryptor.update(plaintext) + encryptor.finalize()
#
# print(f'Ciphertext: {ciphertext}')

# from idea import IDEA
#
# # Define your key (must be 16 bytes long)
# key = b'ThisIsA16ByteKey'
#
# # Create an IDEA instance
# cipher = IDEA(key)
#
# # Define the plaintext (must be a multiple of 8 bytes)
# plaintext = b'YELLOWSUBMARINES'
#
# # Encrypt the plaintext
# ciphertext = cipher.encrypt(plaintext)
#
# print(f'Ciphertext: {ciphertext}')

# from cryptography.hazmat.primitives.asymmetric import ec
# from cryptography.hazmat.primitives import hashes
# from cryptography.hazmat.primitives.kdf.hkdf import HKDF
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.backends import default_backend
# import os
#
# # Generate ECC private key
# private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
#
# # Generate ECC public key
# public_key = private_key.public_key()
#
# # Generate a shared secret
# shared_key = private_key.exchange(ec.ECDH(), public_key)
#
# # Derive a symmetric key from the shared secret
# derived_key = HKDF(
#     algorithm=hashes.SHA256(),
#     length=32,
#     salt=None,
#     info=b'handshake data',
#     backend=default_backend()
# ).derive(shared_key)
#
# # Encrypt data using the derived symmetric key
# plaintext = b'YELLOWSUBMARINES'
# iv = os.urandom(16)
# cipher = Cipher(algorithms.AES(derived_key), modes.CFB(iv), backend=default_backend())
# encryptor = cipher.encryptor()
# ciphertext = encryptor.update(plaintext) + encryptor.finalize()
#
# print(f"Ciphertext: {ciphertext.decode('utf-16')}")

from ecdsa import SigningKey, SECP256k1
from hashlib import sha256

# Generate ECC private key
private_key = SigningKey.generate(curve=SECP256k1)

# Generate ECC public key
public_key = private_key.get_verifying_key()

# Example data to encrypt
data = b'YELLOWSUBMARINES'

# Hash the data
hashed_data = sha256(data).digest()

# Sign the hashed data
signature = private_key.sign(hashed_data)

print(f'Signature: {signature.decode("ascii")}')
