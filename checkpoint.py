from dataclasses import asdict

import torch

from model import GPTConfig, HeLLM
from tokenizer import tokenizer_from_dict


def save_checkpoint(path, model, tokenizer, **extra):
    # Only tensors and plain Python values, so the file loads with weights_only=True.
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.config),
            "tokenizer": tokenizer.to_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = HeLLM(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, tokenizer_from_dict(checkpoint["tokenizer"]), checkpoint
