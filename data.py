from pathlib import Path

import torch


def load_text(num_stories=1000, data_file=None):
    """Text from a local file, or the first num_stories stories of TinyStories."""
    if data_file:
        return Path(data_file).read_text(encoding="utf-8")

    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{num_stories}]")
    return "\n".join(ds["text"])


def split_tokens(ids, val_fraction=0.1):
    data = torch.tensor(ids, dtype=torch.long)
    n_val = int(len(data) * val_fraction)
    return data[:len(data) - n_val], data[len(data) - n_val:]


def get_batch(data, block_size, batch_size, device="cpu"):
    if len(data) <= block_size + 1:
        raise ValueError(f"only {len(data)} tokens, need more than block_size + 1 ({block_size + 1})")
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
