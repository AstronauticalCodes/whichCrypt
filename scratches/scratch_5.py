from Crypto.Cipher import AES, ARC2, ARC4, CAST, DES
from Crypto.Random import get_random_bytes
from Crypto.PublicKey import RSA
import base64
from csv import writer
from random import shuffle
import base64


key = get_random_bytes(16)
cipher = AES.new(key, AES.MODE_EAX)
nonce = cipher.nonce
encryptCypherText, tag = cipher.encrypt_and_digest("linasdfaaasfdsadasfdasdfasdffsadfs fas fsadvasd fa fdsfsads sdsdaf asdfasf sa fsfdasdfasdfse".encode(encoding='utf-8'))
encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
print(encoded_encryptCypherText)
# encryptRows.append([encoded_encryptCypherText, 'AES'])


# key = get_random_bytes(16)
# cipher = CAST.new(key, CAST.MODE_EAX)
# nonce = cipher.nonce
# encryptCypherText = cipher.encrypt("linasdfasdfasfdase".encode(encoding='utf-8'))
# encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
# # encryptRows.append([encoded_encryptCypherText, 'CAST'])
# print(encoded_encryptCypherText)


# key = get_random_bytes(8)
# cipher = DES.new(key, DES.MODE_EAX)
# nonce = cipher.nonce
# encryptCypherText = cipher.encrypt("DSFASDF AFDSA ASDdasf asfdffg a tjkhjvb v FAS DFASSDAFA F".encode(encoding='utf-8'))
# encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
# print(encoded_encryptCypherText)
