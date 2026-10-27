import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import BPETokenizer, CharTokenizer, tokenizer_from_dict

TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside. "
    "One day, Lily saw a big red ball. She ran to the ball and kicked it. "
    "The little dog ran after the ball too. Lily and the dog played all day. "
) * 20


class CharTokenizerTest(unittest.TestCase):
    def test_round_trip(self):
        tok = CharTokenizer.train(TEXT)
        self.assertEqual(tok.decode(tok.encode(TEXT)), TEXT)
        self.assertEqual(tok.vocab_size, len(set(TEXT)))

    def test_unknown_characters_are_skipped(self):
        tok = CharTokenizer.train("abc ")
        self.assertEqual(tok.unknown_chars("abz!"), ["!", "z"])
        self.assertEqual(tok.decode(tok.encode("abz!")), "ab")


class BPETokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.train(TEXT, vocab_size=320)

    def test_vocab_size(self):
        self.assertEqual(self.tok.vocab_size, 320)

    def test_round_trip_including_unseen_unicode(self):
        for text in (TEXT, "Lily saw a café 🐶 and said \"wow!\"\n\nThe end.", ""):
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_compresses_training_text(self):
        self.assertLess(len(self.tok.encode(TEXT)), len(TEXT.encode("utf-8")) / 2)

    def test_common_words_become_single_tokens(self):
        self.assertEqual(len(self.tok.encode(" Lily")), 1)

    def test_dict_round_trip(self):
        loaded = tokenizer_from_dict(self.tok.to_dict())
        self.assertEqual(loaded.encode(TEXT), self.tok.encode(TEXT))

    def test_training_is_deterministic(self):
        again = BPETokenizer.train(TEXT, vocab_size=320)
        self.assertEqual(again.merges, self.tok.merges)

    def test_merges_match_naive_training(self):
        # Retrain the slow way (recount every pair each step) and compare the first merges.
        from collections import Counter

        from tokenizer import CHUNK_PATTERN, merge_pair

        words = Counter(tuple(c.encode("utf-8")) for c in CHUNK_PATTERN.findall(TEXT))
        merges = []
        for new_id in range(256, 276):
            pairs = Counter()
            for word, n in words.items():
                for pair in zip(word, word[1:]):
                    pairs[pair] += n
            best_count = max(pairs.values())
            merges.append({p for p, c in pairs.items() if c == best_count})
            best = self.tok.merges[new_id - 256]
            words = {tuple(merge_pair(list(w), best, new_id)): n for w, n in words.items()}
        for rank, tied in enumerate(merges):
            self.assertIn(self.tok.merges[rank], tied)


if __name__ == "__main__":
    unittest.main()
