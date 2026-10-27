import argparse
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from checkpoint import save_checkpoint
from data import get_batch, load_text, pick_device, split_tokens
from model import GPTConfig, HeLLM
from tokenizer import train_tokenizer

SAMPLE_PROMPTS = ["Once upon a time", "The little dog", "Lily and Tom"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train HeLLM on TinyStories or a local text file.")

    g = p.add_argument_group("data")
    g.add_argument("--num-stories", type=int, default=1000, help="TinyStories stories to train on")
    g.add_argument("--data-file", help="train on this text file instead of TinyStories")
    g.add_argument("--val-fraction", type=float, default=0.1)
    g.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    g.add_argument("--vocab-size", type=int, default=1024, help="BPE vocabulary size")

    g = p.add_argument_group("model")
    g.add_argument("--block-size", type=int, default=128)
    g.add_argument("--n-embd", type=int, default=128)
    g.add_argument("--n-head", type=int, default=4)
    g.add_argument("--n-layer", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--pos-encoding", choices=["rope", "learned"], default="rope")
    g.add_argument("--no-tie-weights", action="store_true")

    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--max-steps", type=int, default=5000)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--min-lr", type=float, default=1e-4)
    g.add_argument("--warmup-steps", type=int, default=200)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--amp", action="store_true", help="bfloat16 mixed precision (CUDA only)")
    g.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    g.add_argument("--seed", type=int, default=1337)

    g = p.add_argument_group("logging")
    g.add_argument("--eval-interval", type=int, default=250)
    g.add_argument("--eval-iters", type=int, default=50)
    g.add_argument("--log-interval", type=int, default=50)
    g.add_argument("--out-dir", default="checkpoints")
    return p.parse_args(argv)


def lr_at(step, args):
    """Linear warmup, then cosine decay from lr down to min_lr."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model, args):
    # Weight decay on matrices only; biases and LayerNorm parameters are left alone.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def estimate_loss(model, splits, args, device, autocast):
    model.eval()
    losses = {}
    for name, data in splits.items():
        total = 0.0
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.block_size, args.batch_size, device)
            with autocast:
                _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / args.eval_iters
    model.train()
    return losses


@torch.no_grad()
def write_samples(model, tokenizer, device, step, path):
    model.eval()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"===== step {step} =====\n")
        for prompt in SAMPLE_PROMPTS:
            ids = tokenizer.encode(prompt)
            if not ids:
                continue
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, 200, temperature=0.8, top_k=50)
            f.write(tokenizer.decode(out[0].tolist()) + "\n---\n")
    model.train()


def plot_losses(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping the loss plot")
        return

    steps = [r["step"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [r["train_loss"] for r in rows], label="train")
    plt.plot(steps, [r["val_loss"] for r in rows], label="validation")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = load_text(args.num_stories, args.data_file)
    started = time.time()
    tokenizer = train_tokenizer(args.tokenizer, text, args.vocab_size)
    ids = tokenizer.encode(text)
    print(f"Tokenizer: {args.tokenizer}, vocabulary size {tokenizer.vocab_size} ({time.time() - started:.1f}s)")
    print(f"Training text: {len(text):,} characters -> {len(ids):,} tokens ({len(text) / len(ids):.2f} chars/token)")

    train_data, val_data = split_tokens(ids, args.val_fraction)
    splits = {"train": train_data, "val": val_data}

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        pos_encoding=args.pos_encoding,
        tie_weights=not args.no_tie_weights,
    )
    model = HeLLM(config).to(device)
    print(f"Parameters: {model.num_parameters():,}  Device: {device}")

    optimizer = make_optimizer(model, args)
    use_amp = args.amp and device == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    extra = {
        "train_args": vars(args),
        "data": {"num_stories": args.num_stories, "data_file": args.data_file, "val_fraction": args.val_fraction},
    }

    metrics_path = out_dir / "metrics.csv"
    samples_path = out_dir / "samples.txt"
    samples_path.write_text("", encoding="utf-8")
    rows = []
    best_val = float("inf")

    model.train()
    started = time.time()
    for step in range(args.max_steps):
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model, splits, args, device, autocast)
            rows.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr})
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_dir / "best.pt", model, tokenizer, step=step, val_loss=best_val, **extra)
            write_samples(model, tokenizer, device, step, samples_path)

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {time.time() - started:.0f}s")

    save_checkpoint(out_dir / "last.pt", model, tokenizer, step=args.max_steps, val_loss=rows[-1]["val_loss"], **extra)

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)
    plot_losses(rows, out_dir / "loss_curve.png")

    print(f"Best validation loss {best_val:.4f}. Checkpoints, metrics and samples are in {out_dir}/")


if __name__ == "__main__":
    main()
