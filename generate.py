import argparse
import sys

import torch

from checkpoint import load_checkpoint
from data import pick_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate text with a trained HeLLM checkpoint.")
    p.add_argument("prompt", nargs="?", help="prompt text; asks interactively if left out")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8, help="0 always picks the most likely token")
    p.add_argument("--top-k", type=int, default=50, help="0 turns it off")
    p.add_argument("--top-p", type=float, default=None, help="nucleus sampling, e.g. 0.9")
    p.add_argument("--no-cache", action="store_true", help="recompute every position at each step")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def generate(model, tokenizer, prompt, args, device):
    unknown = tokenizer.unknown_chars(prompt)
    if unknown:
        print(f"(skipping characters the model never saw: {''.join(unknown)!r})", file=sys.stderr)
    ids = tokenizer.encode(prompt)
    if not ids:
        print("The prompt has no characters the model knows.", file=sys.stderr)
        return

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stream = model.generate_stream(
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
        top_p=args.top_p,
        use_cache=not args.no_cache,
    )

    print(prompt, end="", flush=True)
    new_ids = []
    printed = 0
    for token in stream:
        new_ids.append(token.item())
        text = tokenizer.decode(new_ids)
        # A BPE token can end partway through a multi-byte character; wait for the rest of it.
        safe = text.rstrip("�")
        print(safe[printed:], end="", flush=True)
        printed = len(safe)
    print(tokenizer.decode(new_ids)[printed:])


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    print(
        f"Loaded {args.checkpoint} (step {checkpoint.get('step')}, "
        f"val loss {checkpoint.get('val_loss', float('nan')):.4f}, {model.num_parameters():,} parameters)",
        file=sys.stderr,
    )

    if args.prompt is not None:
        generate(model, tokenizer, args.prompt, args, device)
        return

    while True:
        try:
            prompt = input("\nEnter a prompt (empty to quit): ")
        except EOFError:
            break
        if not prompt:
            break
        generate(model, tokenizer, prompt, args, device)


if __name__ == "__main__":
    main()
