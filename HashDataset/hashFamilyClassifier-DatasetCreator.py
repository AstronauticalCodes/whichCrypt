import csv
from random import randint, shuffle

import numpy as np
from hashlib import sha256, sha224, sha384, sha512, sha3_224, sha3_256, sha3_384, sha3_512, sha1
from hashlib import md5
from hashlib import blake2b, blake2s
from hashlib import shake_256, shake_128
import hashlib

with open("Big-wC.txt", encoding='utf-8') as file:
    data = file.read().split('\n')
    data = [x for x in data if len(x) != 0] 
    data = data[:len(data)//500]

with open("HFC1-mini-TD-wC-SHA-BLAKE.csv", 'w', newline='', encoding='utf-8') as csvFile:
    writer = csv.writer(csvFile)
    features = ["Hash"]
    label = ["Family"]
    colNames = features + label
    writer.writerow(colNames)

    hashRows = []
    maxLen = 0
    for line in data:
        if len(line) > maxLen:
            maxLen = len(line)
        #SHA Family
        hashRows.append([sha1(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        # hashRows.append([sha224(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        hashRows.append([sha256(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        # hashRows.append([sha384(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        hashRows.append([sha512(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        hashRows.append([sha3_224(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        # hashRows.append([sha3_256(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        hashRows.append([sha3_384(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])
        # hashRows.append([sha3_512(line.encode(encoding='utf-8')).hexdigest(), "SHA-Family"])

        #blake Family
        hashRows.append([blake2s(line.encode(encoding='utf-8')).hexdigest(), "Blake-Family"])
        hashRows.append([blake2b(line.encode(encoding='utf-8')).hexdigest(), "Blake-Family"])

        #shake Family
        # hashRows.append([shake_128(line.encode(encoding='utf-8')).hexdigest(randint(1, maxLen)), "Shake-Family"])
        # hashRows.append([shake_256(line.encode(encoding='utf-8')).hexdigest(randint(1, maxLen)), "Shake-Family"])
        #
        # #md5
        # hashRows.append([md5(line.encode(encoding='utf-8')).hexdigest(), "md5"])
        #
        # #RIPEMD-160
        # ripemdHash = hashlib.new("ripemd160")
        # ripemdHash.update(line.encode(encoding='utf-8'))
        # hashRows.append([ripemdHash.hexdigest(), "RIPEMD-160"])
        #
        # #SM3
        # ripemdHash = hashlib.new("sm3")
        # ripemdHash.update(line.encode(encoding='utf-8'))
        # hashRows.append([ripemdHash.hexdigest(), "sm3"])

    shuffle(hashRows)
    writer.writerows(hashRows)
    # print(maxLen)
