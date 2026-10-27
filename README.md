# HeLLM

This project is a small GPT-style language model built from scratch in PyTorch.

The goal of the project is to understand how transformers work by implementing the main parts of a language model directly, including tokenization, embeddings, attention, transformer blocks, training, and text generation.

The model is trained to predict the next token in a sequence.

For example:

```text
Input:
Once there was a little

Target:
girl
```

It uses the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset from Hugging Face.

## What's implemented

**Model** ([model.py](model.py))
- Decoder-only transformer with pre-LayerNorm residual blocks
- Causal multi-head self-attention computed for all heads in one batched operation, using PyTorch's fused `scaled_dot_product_attention` (a manual path returns the attention weights for plotting)
- Rotary position embeddings (RoPE), or learned position embeddings with `--pos-encoding learned`
- GELU feed-forward layers, dropout, and the token embedding shared with the output layer (weight tying)
- GPT-2 style initialization, with the residual projections scaled down by depth

**Tokenizers** ([tokenizer.py](tokenizer.py))
- Character-level tokenizer
- Byte-level BPE written from scratch. Text is split into word-like chunks, then the most frequent adjacent pair is merged repeatedly. Pair counts are updated incrementally, so only the words a merge touches are recounted. Since it works on bytes, it can encode any text, including emoji and characters it never saw.

**Training** ([train.py](train.py))
- Train/validation split, with the validation loss checked during training
- AdamW with weight decay on matrices only, linear warmup then cosine learning-rate decay, and gradient clipping
- Runs on Apple Silicon (MPS), CUDA or CPU; bfloat16 mixed precision on CUDA with `--amp`
- Saves the best checkpoint by validation loss, plus a loss curve, a metrics CSV and samples from fixed prompts at each evaluation

**Inference** ([generate.py](generate.py))
- Streams tokens as they are generated
- Temperature, top-k and top-p (nucleus) sampling
- KV cache: after the prompt, each step runs only the newest token through the model and reuses the keys and values of earlier positions

**Evaluation** ([evaluate.py](evaluate.py), [visualize_attention.py](visualize_attention.py))
- Perplexity and bits per byte on the validation split. Perplexity depends on the tokenizer; bits per byte lets the character and BPE models be compared.
- Heatmaps of each attention head's weights for a prompt

**Tests** ([tests/](tests))
- The causal mask: changing future tokens doesn't change earlier outputs
- KV-cached generation matches uncached generation, including past the context window
- BPE round trips (including unseen Unicode), determinism, and agreement with a slow reference implementation
- Top-k and top-p filtering, the learning-rate schedule, checkpoint round trips, and a short end-to-end training run

## Setup

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train a character-level model on the first 1,000 stories (saved to `checkpoints/`):

```bash
python train.py
```

Train with the BPE tokenizer on more data:

```bash
python train.py --tokenizer bpe --vocab-size 1024 --num-stories 10000 --out-dir checkpoints/bpe
```

`python train.py --help` lists every option (model size, context length, learning rate, and so on). `--data-file` trains on any local text file instead of TinyStories.

Generate text, interactively or from a prompt:

```bash
python generate.py
python generate.py "Once upon a time" --temperature 0.8 --top-p 0.9
```

Evaluate on the validation split and plot attention:

```bash
python evaluate.py --checkpoint checkpoints/best.pt
python visualize_attention.py "Lily saw a big red ball." --layer 0
```

Run the tests:

```bash
python -m unittest discover -s tests -t .
```

## Results

These runs use the default settings: 1,000 stories, a 128-token context, 4 layers, 4 heads, 128-dimensional embeddings, RoPE, and 5,000 steps at batch size 32 on an Apple Silicon GPU.

| Tokenizer | Vocabulary | Parameters | Chars per token | Validation loss | Perplexity | Bits per byte |
|---|---|---|---|---|---|---|
| Character | 75 | 803K | 1.00 | 0.98 | 2.68 | 1.42 |
| BPE | 1,024 | 924K | 3.04 | 2.51 | 12.33 | **1.22** |

The BPE model is better per byte of text (1.22 vs 1.42 bits) because each token covers about 3 characters, so the same 128-token context holds three times as much story. Its training loss (1.52) ends far below its validation loss, though, so with only 1,000 stories it overfits; `--num-stories 10000` or more should help.

Sample from the BPE model (`--temperature 0.8 --top-k 50`):

```text
Once upon a time there was a little girl named Lily. One day, she went to the park with her mom.
her mom said to her mom and dad, "Lily, that's a good day at school! Can we take care of the vest?"

Lily was so happy that she nodded her head.
```

Sample from the character model:

```text
Once upon a time, there was a little boy named Lily. She loved to play with her toys and couldn't
wait to complain with her toys. One day, a little girl named Jane and Jack asked John sight her friends.
```

Inside the 128-token context window, the KV cache makes generation about 2.4x faster on CPU (0.09s vs 0.22s for 120 characters). Past the window, the cache is rebuilt from the last 128 tokens at every step so the output matches uncached generation exactly, so long generations don't get that speedup.
