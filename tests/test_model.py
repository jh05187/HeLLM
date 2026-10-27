import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint import load_checkpoint, save_checkpoint
from model import GPTConfig, HeLLM, KVCache, filter_logits
from tokenizer import CharTokenizer


def tiny_model(pos_encoding="rope", block_size=16):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=11, block_size=block_size, n_embd=16, n_head=2, n_layer=2,
                       dropout=0.0, pos_encoding=pos_encoding)
    return HeLLM(config).eval()


class ModelTest(unittest.TestCase):
    def test_shapes_and_loss(self):
        model = tiny_model()
        idx = torch.randint(0, 11, (3, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (3, 8, 11))
        self.assertTrue(torch.isfinite(loss))

    def test_future_tokens_do_not_change_earlier_logits(self):
        for pos in ("rope", "learned"):
            model = tiny_model(pos)
            a = torch.randint(0, 11, (1, 10))
            b = a.clone()
            b[0, 6:] = (b[0, 6:] + 1) % 11
            self.assertTrue(torch.allclose(model(a)[0][:, :6], model(b)[0][:, :6], atol=1e-6), pos)

    def test_attention_maps_are_causal_and_normalized(self):
        model = tiny_model()
        maps = model.attention_maps(torch.randint(0, 11, (1, 7)))
        self.assertEqual(len(maps), 2)
        for weights in maps:
            self.assertEqual(weights.shape, (1, 2, 7, 7))
            self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(1, 2, 7)))
            self.assertEqual(weights.triu(diagonal=1).abs().sum().item(), 0.0)

    def test_attention_maps_match_fast_path(self):
        model = tiny_model()
        idx = torch.randint(0, 11, (2, 9))
        x_fast, _ = model._hidden(idx)
        x_slow, _ = model._hidden(idx, need_weights=True)
        self.assertTrue(torch.allclose(x_fast, x_slow, atol=1e-5))

    def test_kv_cache_forward_matches_full_forward(self):
        for pos in ("rope", "learned"):
            model = tiny_model(pos)
            idx = torch.randint(0, 11, (2, 12))
            full, _ = model(idx)
            cache = KVCache(model.config.n_layer)
            first, _ = model(idx[:, :5], kv_cache=cache)
            rest, _ = model(idx[:, 5:], kv_cache=cache)
            self.assertTrue(torch.allclose(full, torch.cat((first, rest), dim=1), atol=1e-5), pos)

    def test_cached_generation_matches_uncached_past_the_context_window(self):
        for pos in ("rope", "learned"):
            model = tiny_model(pos, block_size=8)
            idx = torch.randint(0, 11, (2, 3))
            cached = model.generate(idx, 20, temperature=0, use_cache=True)
            uncached = model.generate(idx, 20, temperature=0, use_cache=False)
            self.assertEqual(cached.shape, (2, 23))
            self.assertTrue(torch.equal(cached, uncached), pos)

    def test_weights_are_tied(self):
        model = tiny_model()
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_rejects_sequences_longer_than_block_size(self):
        with self.assertRaises(ValueError):
            tiny_model(block_size=4)(torch.zeros(1, 5, dtype=torch.long))

    def test_checkpoint_round_trip(self):
        model = tiny_model()
        tokenizer = CharTokenizer("abcdefghijk")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            save_checkpoint(path, model, tokenizer, step=3, val_loss=1.5, data={"num_stories": 1})
            loaded, loaded_tokenizer, checkpoint = load_checkpoint(path)
        idx = torch.randint(0, 11, (1, 6))
        self.assertTrue(torch.allclose(model(idx)[0], loaded(idx)[0]))
        self.assertEqual(loaded_tokenizer.encode("cab"), tokenizer.encode("cab"))
        self.assertEqual(checkpoint["step"], 3)


class SamplingTest(unittest.TestCase):
    def test_top_k_keeps_k_largest(self):
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        kept = torch.isfinite(filter_logits(logits, top_k=2))
        self.assertEqual(kept.tolist(), [[False, True, False, True]])

    def test_top_p_keeps_smallest_set_covering_p(self):
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
        kept = torch.isfinite(filter_logits(probs.log(), top_p=0.7))
        self.assertEqual(kept.tolist(), [[True, True, False, False]])

    def test_top_p_always_keeps_the_most_likely_token(self):
        probs = torch.tensor([[0.1, 0.9]])
        kept = torch.isfinite(filter_logits(probs.log(), top_p=0.01))
        self.assertEqual(kept.tolist(), [[False, True]])


if __name__ == "__main__":
    unittest.main()
