'''seq2seq model for recipe title generation based on the ingredient input'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# the encoder reads the input ingredient sequence and compresses it into a hidden state (summarize input ingredients)
class Encoder(nn.Module):
    def __init__(self, n_vocab, emb, hidden, pad_idx, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(n_vocab, emb, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src, src_lens):
        """Pack padded inputs so the SEQ2SEQ ignores pad tokens efficiently."""
        x = self.embed(src)
        packed = pack_padded_sequence(x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return h


# the decoder takes the encoder's hidden state and generates the output title token by token
# (“given what has been generated so far, what word should come next in the title?”)
class Decoder(nn.Module):
    def __init__(self, n_vocab, emb, hidden, pad_idx, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers
        self.embed = nn.Embedding(n_vocab, emb, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden, n_vocab)

    def forward(self, tgt, hidden):
        """Run one or more decoding steps and project to vocabulary logits."""
        x = self.embed(tgt)
        o, h = self.rnn(x, hidden)
        return self.out(o), h


# Thin wrapper that connects the encoder and decoder modules
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt_in):
        """Encode the source once, then decode the teacher-forced target input."""
        enc_h = self.encoder(src, src_lens)
        logits, _ = self.decoder(tgt_in, enc_h)
        return logits
