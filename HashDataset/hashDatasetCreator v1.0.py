import csv
import hashlib
from random import shuffle, randint

ciphers = ['sha256', 'shake256', 'md5']

with open('conversationText.txt', encoding='utf-8') as file:
    data = file.read().split('\n')

with open('newtxt.txt', 'w', encoding='utf-8') as file:
    lst = []
    for x in data:
        lst.append(x + '\n')

    file.write(''.join(lst))

with open('training_dataset_whichCrypt.csv', 'w', newline='', encoding='utf-8') as dataset:
    writer = csv.writer(dataset, delimiter=',')

    features = ['Cipher Text', 'Cipher', 'length']

    writer.writerow(features)

    # for idx in range(3):
    #     start_idx = (idx * len(data)// 3)
    #     end_idx = ((idx + 1) * len(data)//3)
    #
    #     for line in data[start_idx: end_idx]:
    #         pass

    HashRows = []
    maxLen = 0
    for line in data:
        if len(line) > maxLen:
            maxLen = len(line)
        hash = hashlib.sha256(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'sha256', len(hash)])
        # writer.writerow([line, hash, 'sha256', len(hash)])

    for line in data:
        if len(line) > maxLen:
            maxLen = len(line)
        hash = hashlib.sha256(line.encode(encoding='utf-8')).hexdigest()
        HashRows.append([hash, 'md5', len(hash)])
        # writer.writerow([line, hash, 'md5', len(hash)])

    for line in data:
        if len(line) > maxLen:
            maxLen = len(line)
        hash = hashlib.shake_256(line.encode(encoding='utf-8')).hexdigest(randint(1, maxLen))
        HashRows.append([hash, 'shake256', len(hash)])
        # writer.writerow([line, hash, 'sha512', len(hash)])

    shuffle(HashRows)

    writer.writerows(HashRows)
