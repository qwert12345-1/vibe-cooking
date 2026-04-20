"""Simple vocabulary wrapper for token/id conversion in training and inference."""

from collections import Counter


class Vocab:
    """Stores token-index mappings plus helpers for encoding and decoding."""

    # defines special tokens: PAD (padding), UNK (unknown), SOS (start of sequence), EOS (end of sequence)
    PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"

    # stoi: string to index, itos: index to string
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = []

    @classmethod
    def from_itos(cls, itos):
        """Rebuild a vocabulary from a saved index-to-string list."""
        v = cls(min_freq=1)
        v.itos = list(itos)
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v

    # count token frequencies and builds the vocabulary
    def build(self, sequences):
        cnt = Counter()
        for seq in sequences:
            cnt.update(seq)
        specials = [self.PAD, self.UNK, self.SOS, self.EOS]
        self.itos = specials + [w for w, c in cnt.items() if c >= self.min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    # map tokens to ids, optionally wrapping with SOS/EOS
    # e.g. ["i", "have", "chicken", "rice", "and", "garlic"] -> [1, 2, 3, 4, 5, 6] (might not be exact)
    def encode(self, tokens, add_sos_eos=False):
        unk = self.stoi[self.UNK]
        ids = [self.stoi.get(t, unk) for t in tokens]
        if add_sos_eos:
            ids = [self.stoi[self.SOS]] + ids + [self.stoi[self.EOS]]
        return ids

    # map ids back to text while optionally hiding special symbols
    # e.g. [1, 2, 3, 4, 5, 6] -> ["i", "have", "chicken", "rice", "and", "garlic"] (might not be exact)
    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            w = self.itos[i]
            if skip_special and w in (self.PAD, self.UNK, self.SOS, self.EOS):
                continue
            if w == self.EOS:
                break
            out.append(w)
        return " ".join(out)
