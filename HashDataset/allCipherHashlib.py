import hashlib
#
# def hello():
#     print(hashlib.algorithms_guaranteed)
#     encoded = 'Hello'.encode(encoding='utf-8')
#     blake2s = hashlib.blake2s(encoded).hexdigest()
#     sha3_384 = hashlib.sha3_384(encoded).hexdigest()
#     sha3_224 = hashlib.sha3_224(encoded).hexdigest()
#     sha256 = hashlib.sha256(encoded).hexdigest()
#     sha224 = hashlib.sha224(encoded).hexdigest()
#     sha3_256 = hashlib.sha3_256(encoded).hexdigest()
#     shake_256 = hashlib.shake_256(encoded).hexdigest(10)
#     blake2b = hashlib.blake2b(encoded).hexdigest()
#     shake_128 = hashlib.shake_128(encoded)
#     sha1 = hashlib.sha1(encoded).hexdigest()
#     sha512 = hashlib.sha512(encoded).hexdigest()
#     md5 = hashlib.md5(encoded).hexdigest()
#     sha384 = hashlib.sha384(encoded).hexdigest()
#     sha3_512 = hashlib.sha3_512(encoded).hexdigest()
#
#     print(len(blake2s), blake2s)
#     print()
#     print(len(shake_256), shake_256)
#
#
#
#
# shake_256 = hashlib.shake_256('ANY THING'.encode(encoding='utf-8')).hexdigest(10)
# print(f'1. len = {len(shake_256)}, hash = {shake_256}')
#
# shake_256 = hashlib.shake_256('ANY THING'.encode(encoding='utf-8')).hexdigest(15)
# print(f'2. len = {len(shake_256)}, hash = {shake_256}')
#
# shake_256 = hashlib.shake_256('ANY THING'.encode(encoding='utf-8')).hexdigest(20)
# print(f'3. len = {len(shake_256)}, hash = {shake_256}')
#
# shake_256 = hashlib.shake_256('ANY THING'.encode(encoding='utf-8')).hexdigest(40)
# print(f'4. len = {len(shake_256)}, hash = {shake_256}')
#
# shake_256 = hashlib.shake_256('ANY THING'.encode(encoding='utf-8')).hexdigest(220)
# print(f'5. len = {len(shake_256)}, hash = {shake_256}')
#
# shake_256 = hashlib.shake_256('ANY THING'.encode(encoding='utf-8')).hexdigest(1024)
# print(f'6. len = {len(shake_256)}, hash = {shake_256}')

# with open('BigAF - Copy.txt', 'w', encoding='utf-8') as file:
#     with open('Big-wC.txt', encoding='utf-8') as file2:
#         file_data = file2.read()
#         file.write(file_data[:(len(file_data)//2)])

# with open('Big-wC.txt', encoding='utf-8') as file1:
#     with open('BigAF - Copy.txt', encoding='utf-8') as file2:
#         print(len(file1.read()), len(file2.read()))

print(hashlib.sha224("HADLFADF".encode(encoding='utf-8')).hexdigest())
