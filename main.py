import torch
import torch.nn as nn
import torch.nn.functional as F

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

data = torch.tensor(encode(text), dtype=torch.long)

print("Full data:")
print(data)
print("Shape:", data.shape)

block_size = 8

def get_batch():
    start = torch.randint(
        0,
        len(data) - block_size - 1,
        (1,)
    ).item()

    x = data[start:start + block_size]
    y = data[start + 1:start + block_size + 1]

    return x, y

x, y = get_batch()

print("x:", x)
print("y:", y)

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]

    print(
        f"input: '{decode(context.tolist())}'"
        f" -> target: '{decode([target.item()])}'"
    )

class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = TinyLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)

for step in range(1000):
    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Step:", step, "Loss:", loss.item())

x, y = get_batch()
logits, loss = model(x, y)

print("Final loss:", loss.item())
print("Logits shape:", logits.shape)