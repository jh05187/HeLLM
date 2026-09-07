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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(32, head_size, bias=False)
        self.query = nn.Linear(32, head_size, bias=False)
        self.value = nn.Linear(32, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1)

        weights = weights / (k.shape[-1] ** 0.5)

        weights = weights.masked_fill(
            self.tril[:x.shape[0], :x.shape[0]] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        v = self.value(x)

        out = weights @ v

        return out

class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, 32)
        self.position_embedding_table = nn.Embedding(block_size, 32)

        self.attention_head = Head(32)

        self.lm_head = nn.Linear(32, vocab_size)

    def forward(self, idx, targets=None):
        token_embeddings = self.token_embedding_table(idx)

        positions = torch.arange(len(idx))
        position_embeddings = self.position_embedding_table(positions)

        x = token_embeddings + position_embeddings

        x = self.attention_head(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context = idx[-block_size:]

            logits, loss = self(idx_context)

            logits = logits[-1]

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat((idx, next_token))

        retu

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

start = torch.tensor([stoi["h"]], dtype=torch.long)

generated = model.generate(
    start,
    max_new_tokens=50
)

print("Generated text:")
print(decode(generated.tolist()))