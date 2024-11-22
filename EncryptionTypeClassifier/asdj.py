from random import shuffle
import random
import string

def generate_random_line():
    words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit']
    line_length = random.randint(5, 30)  # Random line length between 5 and 15 words
    return ' '.join(random.choices(words, k=line_length))

def generate_unique_lines(num_lines):
    unique_lines = set()
    while len(unique_lines) < num_lines:
        unique_lines.add(generate_random_line())
    return list(unique_lines)

# Generate 50,000 unique lines
lines = generate_unique_lines(50000)

# Save to a file
with open('unique_lines.txt', 'w') as file:
    for line in lines:
        file.write(line + '\n')

print("Generated 50,000 unique lines and saved to 'unique_lines.txt'")


with open("evenMoreBigAF.txt", encoding='utf-8') as f:
    data = f.read().split("\n")
    data = [c for c in data if len(c) > 0]
    shuffle(data)

    with open("Bigger-wC.txt", 'w', encoding='utf-8') as w:
        w.write('\n'.join(data))
