import csv
import hashlib
from random import shuffle, randint

with open('Big-wC.txt', encoding='utf-8') as file:
    data = file.read().split('\n')


with open('B1-TD-wC.csv', 'w', newline='', encoding='utf-8') as dataset:
    writer = csv.writer(dataset, delimiter=',')

    features = ['Cipher Text', 'Cipher']

    writer.writerow(features)

    HashRows = []
    maxLen = 0
    for line in data:
        hash = hashlib.sha256(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha256'])
        # writer.writerow([line, hash, 'sha256', len(hash)])

    for line in data:
        hash = hashlib.sha3_256(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha3_256'])
        # writer.writerow([line, hash, 'md5', len(hash)])

    for line in data:
        hash = hashlib.md5(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'md5'])

    for line in data:
        hash = hashlib.blake2b(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'blake_2b'])

    for line in data:
        hash = hashlib.sha512(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha512'])

    for line in data:
        hash = hashlib.sha3_384(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha3_384'])

    for line in data:
        hash = hashlib.sha1(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha1'])
        # writer.writerow([line, hash, 'md5', len(hash)])

    # for line in data:
    #     if len(line) > maxLen:
    #         maxLen = len(line)
    #     hash = hashlib.shake_256(line.encode(encoding='utf-8')).hexdigest(randint(1, maxLen))
    #     HashRows.append([hash, 'shake256', len(hash)])
    #     # writer.writerow([line, hash, 'sha512', len(hash)])

    shuffle(HashRows)

    writer.writerows(HashRows)
