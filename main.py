import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

batch_size = 32
block_size = 64
n_embd = 64
n_head = 4
n_layer = 4
learning_rate = 0.003
max_steps = 5000

ds = load_dataset("roneneldan/TinyStories", split="train")

text = "\n".join(ds["text"][:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Vocabulary size:", vocab_size)
print("Training characters:", len(text))

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(numbers):
    return "".join(itos[i] for i in numbers)

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

def get_batch():
    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[start:start + block_size]
        for start in starts
    ])

    y = torch.stack([
        data[start + 1:start + block_size + 1]
        for start in starts
    ])

    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1)

        weights = weights / (
            k.shape[-1] ** 0.5
        )

        weights = weights.masked_fill(
            self.tril[
                :x.shape[1],
                :x.shape[1]
            ] == 0,
            float("-inf")
        )

        weights = F.softmax(
            weights,
            dim=-1
        )

        v = self.value(x)

        out = weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            n_embd,
            n_embd
        )

    def forward(self, x):
        out = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        return self.projection(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                n_embd,
                4 * n_embd
            ),
            nn.ReLU(),
            nn.Linear(
                4 * n_embd,
                n_embd
            )
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

        return x

class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):
        token_embeddings = (
            self.token_embedding_table(idx)
        )

        positions = torch.arange(
            idx.shape[1],
            device=idx.device
        )

        position_embeddings = (
            self.position_embedding_table(
                positions
            )
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape

            logits_for_loss = logits.reshape(
                B * T,
                C
            )

            targets_for_loss = targets.reshape(
                B * T
            )

            loss = F.cross_entropy(
                logits_for_loss,
                targets_for_loss
            )

        return logits, loss

    def generate(
        self,
        idx,
        max_new_tokens
    ):
        for _ in range(max_new_tokens):

            idx_context = idx[
                :,
                -block_size:
            ]

            logits, loss = self(
                idx_context
            )

            logits = logits[
                :,
                -1,
                :
            ]

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (
                    idx,
                    next_token
                ),
                dim=1
            )

        return idx

model = TinyLanguageModel()

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{parameter_count:,}"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

for step in range(max_steps):
    x, y = get_batch()

    logits, loss = model(
        x,
        y
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(
            "Step:",
            step,
            "Loss:",
            loss.item()
        )

start = torch.tensor(
    [[stoi["h"]]],
    dtype=torch.long
)

generated = model.generate(
    start,
    max_new_tokens=300
)

print("\nGenerated text:")
print(
    decode(
        generated[0].tolist()
    )
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer
    },
    "tiny_model.pth"
)

print("Model saved.")