import argparse

import torch

from checkpoint import load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot each head's attention weights for a prompt.")
    p.add_argument("prompt", nargs="?", default="Once upon a time there was a little girl.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--layer", type=int, default=-1, help="layer to plot; -1 is the last one")
    p.add_argument("--out", default="attention.png")
    return p.parse_args(argv)


def token_labels(tokenizer, ids):
    labels = []
    for i in ids:
        label = tokenizer.decode([i])
        labels.append(label.replace(" ", "␣").replace("\n", "\\n"))
    return labels


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, "cpu")
    ids = tokenizer.encode(args.prompt)[-model.config.block_size:]
    attention = model.attention_maps(torch.tensor([ids], dtype=torch.long))
    layer = args.layer % model.config.n_layer
    weights = attention[layer][0]  # (n_head, T, T)
    labels = token_labels(tokenizer, ids)

    n_head = weights.shape[0]
    cols = min(n_head, 2)
    rows = (n_head + cols - 1) // cols
    size = max(4, 0.25 * len(ids))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for h, ax in enumerate(axes.flat):
        if h >= n_head:
            ax.axis("off")
            continue
        ax.imshow(weights[h].numpy(), cmap="viridis", vmin=0.0)
        ax.set_title(f"layer {layer}, head {h}")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("attended-to token")
        ax.set_ylabel("query token")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
