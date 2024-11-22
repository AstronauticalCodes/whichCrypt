import csv
import hashlib
from random import shuffle, randint
from Crypto.Hash import MD4, RIPEMD, RIPEMD160, TupleHash128
from Crypto.Hash import TupleHash256, TurboSHAKE128, TurboSHAKE256
from Crypto.Hash import KangarooTwelve, KMAC128, KMAC256
from Crypto.Hash import HMAC, MD2, CMAC
from Crypto.Hash import SHA, Poly1305

with open('BigAF.txt', encoding='utf-8') as file:
    data = file.read().split('\n')

# print(hashlib.algorithms_available)

with open('P22-TD-wC.csv', 'w', newline='', encoding='utf-8') as dataset:
    writer = csv.writer(dataset, delimiter=',')

    features = ['Cipher Text', 'Cipher']

    writer.writerow(features)

    HashRows = []
    maxLen = 0
    # for line in data:
    #     hash = hashlib.sha256(line.encode(encoding='utf-8')).hexdigest()
    #     HashRows.append([hash, 'sha256'])
    # #
    # for line in data:
    #     hash = hashlib.sha224(line.encode(encoding='utf-8')).hexdigest()
    #     HashRows.append([hash, 'sha224'])
    #
    # for line in data:
    #     hash = hashlib.sha384(line.encode(encoding='utf-8')).hexdigest()
    #     HashRows.append([hash, 'sha384'])
    # #
    # for line in data:
    #     hash = hashlib.sha512(line.encode(encoding='utf-8')).hexdigest()
    #     HashRows.append([hash, 'sha512'])

    # for line in data:
    #     hash = MD4.new(line.encode(encoding='utf-8')).hexdigest()
    #     HashRows.append([hash, 'md4'])

    for line in data:
        hash = hashlib.md5(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'md5'])

    for line in data:
        hash = hashlib.sha1(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha1'])

    for line in data:
        hash = hashlib.sha3_224(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha3_224'])

    for line in data:
        hash = hashlib.sha3_256(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha3_256'])

    for line in data:
        hash = hashlib.sha3_384(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha3_384'])

    for line in data:
        hash = hashlib.sha3_512(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha3_512'])




    # for line in data:
    #     hash = hashlib.blake2b(line.encode(encoding='utf-8')).hexdigest()
    #     HashRows.append([hash, 'blake2b'])

    # for line in data:
    #     hash = hashlib.blake2s(line.encode(encoding='utf-8')).hexdigest()
    #     HashRows.append([hash, 'blake2s'])


    # for line in data:
    #     if len(line) > maxLen:
    #         maxLen = len(line)
    #     hash = hashlib.shake_256(line.encode(encoding='utf-8')).hexdigest(randint(1, maxLen))
    #     HashRows.append([hash, 'shake_256'])
    #
    # for line in data:
    #     if len(line) > maxLen:
    #         maxLen = len(line)
    #     hash = hashlib.shake_128(line.encode(encoding='utf-8')).hexdigest(randint(1, maxLen))
    #     HashRows.append([hash, 'shake_128'])

    shuffle(HashRows)

    writer.writerows(HashRows)
