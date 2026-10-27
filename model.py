import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    pos_encoding: str = "rope"  # "rope" or "learned"
    tie_weights: bool = True

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.pos_encoding not in ("rope", "learned"):
            raise ValueError("pos_encoding must be 'rope' or 'learned'")
        if self.pos_encoding == "rope" and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE needs an even head size")


class KVCache:
    """Keys and values from earlier positions, one entry per layer."""

    def __init__(self, n_layer):
        self.keys = [None] * n_layer
        self.values = [None] * n_layer

    @property
    def length(self):
        return 0 if self.keys[0] is None else self.keys[0].shape[2]

    def update(self, layer, k, v):
        if self.keys[layer] is not None:
            k = torch.cat((self.keys[layer], k), dim=2)
            v = torch.cat((self.values[layer], v), dim=2)
        self.keys[layer] = k
        self.values[layer] = v
        return k, v


def rope_tables(block_size, head_size, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
    angles = torch.outer(torch.arange(block_size).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_size); cos/sin: (T, head_size / 2)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def causal_mask(query_len, key_len, device):
    # Query i sits at absolute position (key_len - query_len + i) and may see keys up to it.
    past = key_len - query_len
    rows = torch.arange(query_len, device=device).unsqueeze(1) + past
    cols = torch.arange(key_len, device=device).unsqueeze(0)
    return cols <= rows


class CausalSelfAttention(nn.Module):
    """All heads at once: one projection for q, k and v, then reshape into heads."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.pos_encoding == "rope"

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            cos, sin = rope_tables(config.block_size, self.head_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, layer=0, kv_cache=None, past=0, need_weights=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_size).transpose(1, 2) for t in (q, k, v))

        if self.use_rope:
            cos = self.rope_cos[past:past + T]
            sin = self.rope_sin[past:past + T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer, k, v)

        key_len = k.shape[2]
        weights = None
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            scores = scores.masked_fill(~causal_mask(T, key_len, x.device), float("-inf"))
            weights = F.softmax(scores, dim=-1)
            out = F.dropout(weights, self.dropout, self.training) @ v
        else:
            dropout_p = self.dropout if self.training else 0.0
            if key_len == T:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            else:
                mask = causal_mask(T, key_len, x.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.projection(out)), weights


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.projection = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.projection(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.sa = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x, layer=0, kv_cache=None, past=0, need_weights=False):
        attn_out, weights = self.sa(self.ln1(x), layer, kv_cache, past, need_weights)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, weights


def filter_logits(logits, top_k=None, top_p=None):
    """Set logits outside the top-k tokens or the top-p probability mass to -inf."""
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        # Drop a token when the tokens ranked above it already cover top_p; the first always stays.
        remove = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return logits


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = filter_logits(logits / temperature, top_k, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


class HeLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Scale down the layers that write into the residual stream so its variance
        # does not grow with depth (as in GPT-2).
        for name, p in self.named_parameters():
            if name.endswith("projection.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def _hidden(self, idx, kv_cache=None, need_weights=False):
        B, T = idx.shape
        past = kv_cache.length if kv_cache is not None else 0
        if past + T > self.config.block_size:
            raise ValueError(f"sequence of {past + T} tokens exceeds block_size {self.config.block_size}")

        x = self.token_embedding(idx)
        if self.config.pos_encoding == "learned":
            x = x + self.position_embedding(torch.arange(past, past + T, device=idx.device))
        x = self.dropout(x)

        attention = []
        for layer, block in enumerate(self.blocks):
            x, weights = block(x, layer, kv_cache, past, need_weights)
            attention.append(weights)
        return self.ln_f(x), attention

    def forward(self, idx, targets=None, kv_cache=None):
        x, _ = self._hidden(idx, kv_cache)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def attention_maps(self, idx):
        """Attention weights for every layer, each shaped (B, n_head, T, T)."""
        _, attention = self._hidden(idx[:, -self.config.block_size:], need_weights=True)
        return attention

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True):
        """Yield one new token id tensor of shape (B, 1) at a time.

        With use_cache, each step only runs the newest token through the model and reuses
        the keys and values of earlier positions. Once the context window is full, the
        cache is rebuilt from the last block_size tokens, which matches the uncached output.
        """
        block_size = self.config.block_size
        cache = None
        for _ in range(max_new_tokens):
            if use_cache and cache is not None and cache.length < block_size:
                x, _ = self._hidden(idx[:, -1:], cache)
            else:
                cache = KVCache(self.config.n_layer) if use_cache else None
                x, _ = self._hidden(idx[:, -block_size:], cache)

            logits = self.lm_head(x[:, -1, :])
            next_token = sample_next_token(logits, temperature, top_k, top_p)
            idx = torch.cat((idx, next_token), dim=1)
            yield next_token

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, **sampling):
        new_tokens = list(self.generate_stream(idx, max_new_tokens, **sampling))
        return torch.cat([idx] + new_tokens, dim=1)
