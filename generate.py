import torch
import torch.nn as nn
import torch.nn.functional as F

checkpoint = torch.load(
    "tiny_model.pth",
    map_location="cpu",
    weights_only=False
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
n_embd = checkpoint["n_embd"]
n_head = checkpoint["n_head"]
n_layer = checkpoint["n_layer"]

def encode(s):
    return [stoi[c] for c in s]

def decode(numbers):
    return "".join(itos[i] for i in numbers)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

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
            self.tril[:x.shape[1], :x.shape[1]] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        v = self.value(x)

        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return self.projection(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
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
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

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

    def forward(self, idx):
        token_embeddings = self.token_embedding_table(idx)

        positions = torch.arange(
            idx.shape[1],
            device=idx.device
        )

        position_embeddings = self.position_embedding_table(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.blocks(x)
        x = self.ln_f(x)

        return self.lm_head(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -block_size:]

            logits = self(idx_context)

            logits = logits[:, -1, :]

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx

model = TinyLanguageModel()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

prompt = input("Enter a prompt: ")

encoded_prompt = encode(prompt)

start = torch.tensor(
    [encoded_prompt],
    dtype=torch.long
)

with torch.no_grad():
    generated = model.generate(
        start,
        max_new_tokens=300
    )

print()
print(decode(generated[0].tolist()))