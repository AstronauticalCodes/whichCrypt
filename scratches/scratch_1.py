# import hashlib
#
# print(hashlib.sha3_512("dc".encode(encoding='utf-8')).hexdigest())


# import json
# with open("pycssfasd.json", 'w') as file:
#     json.dump({'hello': ['helllo']}, file)


# from Crypto.Cipher import AES
# from Crypto.Random import get_random_bytes
# import base64
#
# key = get_random_bytes(16)
# cipher = AES.new(key, AES.MODE_EAX)
# nonce = cipher.nonce
# encryptCypherText, tag = cipher.encrypt_and_digest("line".encode(encoding='utf-8'))
# encoded_encryptCypherText = base64.b64encode(encryptCypherText).decode('utf-8')
#
# print(encoded_encryptCypherText)

from Crypto.PublicKey import RSA
from sympy import gcd
from math import isqrt


def is_prime(n):
    # Use a simple method for checking primality
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def check_rsa_key(public_key):
    n = public_key.n
    e = public_key.ecl

    # Check if n can be factored into two large primes (simplified)
    for p in range(2, isqrt(n)):
        if n % p == 0:
            q = n // p
            if is_prime(p) and is_prime(q):
                print(f"RSA Key detected with primes p={p}, q={q}")
                phi_n = (p - 1) * (q - 1)
                if 2 < e < phi_n and gcd(e, phi_n) == 1:
                    print("Public key exponent e satisfies RSA conditions")
                    return True
                else:
                    print("Public key exponent e does not satisfy RSA conditions")
                    return False
    return False


# Load the public key (RSA example)
key = RSA.import_key(open('public.pem').read())
# key = '''-----BEGIN PUBLIC KEY-----
# MIIBITANBgkqhkiG9w0BAQEFAAOCAQ4AMIIBCQKCAQBdoenuycmNhCeBiqMHLD4U
# C3pZWTG9aPbjjrKx8WC902MnmrPLmO03mxEK583vXLHCBj0go8O03aEgOxSfeWvn
# aFtsps3FEN/MzBROv9lZJb1PV4DRORm9RN7zIQCTxweSZSXCuXoFTIm2Tqs91+PT
# f7Ao9+0mZhQUZeDorSxQcIfbiusqa7fp186uv1+UruH8QuuZxmtO15yv0WWs4Ewq
# 4KQB1mAcaSYeV93SIWOoY+4ky4w3GSXSAIwOt1WPClB6XfwHy2snEjQDOEX6atjP
# saY9p1E7LLPCBVqjNgKhj/8NxrfgvsZ8iwSwBMiYAxqEe2KUKDUKNGnkJbXuJwEt
# AgMBAAE=
# -----END PUBLIC KEY-----'''
if check_rsa_key(key):
    print("This key uses RSA encryption.")
else:
    print("This key does not use RSA encryption.")
