import argparse
import math

import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint
from data import load_text, pick_device, split_tokens


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Perplexity and bits per byte on the validation split.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data-file", help="evaluate on this text file instead of the training data's validation split")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    """Total negative log-likelihood (nats) and token count over non-overlapping windows."""
    block_size = model.config.block_size
    starts = list(range(0, len(data) - 1, block_size))
    full = [s for s in starts if s + block_size + 1 <= len(data)]
    # Full windows go through in batches; the shorter leftover window at the end goes alone.
    batches = [full[i:i + batch_size] for i in range(0, len(full), batch_size)]
    batches += [[s] for s in starts if s not in full]

    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        n = min(block_size, len(data) - 1 - batch[0])
        x = torch.stack([data[s:s + n] for s in batch]).to(device)
        y = torch.stack([data[s + 1:s + n + 1] for s in batch]).to(device)
        logits, _ = model(x)
        total_nll += F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="sum").item()
        total_tokens += y.numel()
    return total_nll, total_tokens


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)

    if args.data_file:
        data = torch.tensor(tokenizer.encode(load_text(data_file=args.data_file)), dtype=torch.long)
    else:
        settings = checkpoint["data"]
        text = load_text(settings["num_stories"], settings["data_file"])
        _, data = split_tokens(tokenizer.encode(text), settings["val_fraction"])

    nll, tokens = evaluate(model, data, args.batch_size, device)
    n_bytes = len(tokenizer.decode(data.tolist()).encode("utf-8"))

    print(f"Tokens evaluated:  {tokens:,}")
    print(f"Loss per token:    {nll / tokens:.4f}")
    print(f"Perplexity:        {math.exp(nll / tokens):.2f}")
    # Perplexity depends on the tokenizer; bits per byte lets char and BPE models be compared.
    print(f"Bits per byte:     {nll / math.log(2) / n_bytes:.4f}")


if __name__ == "__main__":
    main()
