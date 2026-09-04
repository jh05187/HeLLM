import torch

with open("training_data.txt", "r") as f:
    text = f.read()

print(text)

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(chars)
print("Vocabulary size:", vocab_size)

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]


def decode(numbers):
    return "".join(itos[i] for i in numbers)

encoded = encode("hello")

print("Encoded:", encoded)
print("Decoded:", decode(encoded))