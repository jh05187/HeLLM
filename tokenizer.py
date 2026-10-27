import re
from collections import Counter, defaultdict

# Split text into word-like chunks before BPE so merges never cross a word boundary.
# A leading space stays attached to the word that follows it, as in GPT-2.
CHUNK_PATTERN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+")


class CharTokenizer:
    """One token per character seen in the training text."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def train(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def unknown_chars(self, text):
        return sorted(set(c for c in text if c not in self.stoi))

    def encode(self, text):
        # Characters the model never saw have no id, so they are left out.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def to_dict(self):
        return {"type": "char", "chars": "".join(self.chars)}


def merge_pair(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level byte pair encoding.

    Every string is first turned into UTF-8 bytes (ids 0-255), so any text can be
    encoded. Training then repeatedly merges the most frequent adjacent pair of ids
    into a new id until the vocabulary reaches the requested size.
    """

    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for new_id, (a, b) in enumerate(self.merges, start=256):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        chunk_counts = Counter(CHUNK_PATTERN.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # Count every adjacent pair once up front, then update only the words a merge touches.
        pair_counts = Counter()
        words_with_pair = defaultdict(set)
        for wi, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[wi]
                words_with_pair[pair].add(wi)

        merges = []
        for new_id in range(256, vocab_size):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            merges.append(best)

            for wi in list(words_with_pair[best]):
                word, freq = words[wi], freqs[wi]
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    words_with_pair[pair].discard(wi)
                word = merge_pair(word, best, new_id)
                words[wi] = word
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += freq
                    words_with_pair[pair].add(wi)

        return cls(merges)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def unknown_chars(self, text):
        return []

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge first, the same order training used.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge_pair(ids, pair, 256 + self.ranks[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for chunk in CHUNK_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def to_dict(self):
        return {"type": "bpe", "merges": [list(pair) for pair in self.merges]}


def train_tokenizer(kind, text, vocab_size=None):
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "bpe":
        return BPETokenizer.train(text, vocab_size)
    raise ValueError(f"unknown tokenizer {kind!r}")


def tokenizer_from_dict(data):
    if data["type"] == "char":
        return CharTokenizer(data["chars"])
    if data["type"] == "bpe":
        return BPETokenizer(data["merges"])
    raise ValueError(f"unknown tokenizer {data['type']!r}")
