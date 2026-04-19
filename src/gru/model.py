"""Minimal GRU encoder-decoder model used for recipe title generation."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    """Encode ingredient tokens into a final recurrent hidden state."""

    def __init__(self, n_vocab, emb, hidden, pad_idx, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(n_vocab, emb, padding_idx=pad_idx)
        self.gru = nn.GRU(
            emb, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src, src_lens):
        """Pack padded inputs so the GRU ignores pad tokens efficiently."""
        x = self.embed(src)
        packed = pack_padded_sequence(x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h


class Decoder(nn.Module):
    """Decode recipe-title tokens from the encoder hidden state."""

    def __init__(self, n_vocab, emb, hidden, pad_idx, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers
        self.embed = nn.Embedding(n_vocab, emb, padding_idx=pad_idx)
        self.gru = nn.GRU(
            emb, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden, n_vocab)

    def forward(self, tgt, hidden):
        """Run one or more decoding steps and project to vocabulary logits."""
        x = self.embed(tgt)
        o, h = self.gru(x, hidden)
        return self.out(o), h


class Seq2Seq(nn.Module):
    """Thin wrapper that connects the encoder and decoder modules."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt_in):
        """Encode the source once, then decode the teacher-forced target input."""
        enc_h = self.encoder(src, src_lens)
        logits, _ = self.decoder(tgt_in, enc_h)
        return logits
