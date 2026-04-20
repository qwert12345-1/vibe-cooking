"""Inference helpers for greedy recipe-title generation from ingredients.

Adapted from the original vibe-cooking-main project. Imports rewritten to
use this sub-package's relative modules so the sub-package is self-contained.
"""
from __future__ import annotations

import re

import torch

from .model import Decoder, Encoder, Seq2Seq
from .tokens import tokenize
from .vocab import Vocab

# ensures the input format looks like training data
# e.g. if the user types "chicken rice garlic", it becomes "i have chicken rice garlic"
def normalize_user_text(s: str) -> str:
    s = (s or "").lower().strip()
    if not s.startswith("i have"):
        s = "i have " + re.sub(r"^i\s+have\s+", "", s, flags=re.I)
    return s


# removes useless stuff like repeated spaces, <unk>, and weird leading contractions like "'s, 'll, etc."
# so the final title is cleaner for display
# e.g. if the model outputs "i have chicken rice garlic <eos>", it becomes "i have chicken rice garlic"
def clean_generated_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace(Vocab.UNK, " ")).strip()
    text = re.sub(
        r"^(?:['`’]?s|['`’]?d|['`’]?ll|['`’]?re|['`’]?ve|['`’]?m)\b[\s,.:;-]*",
        "",
        text,
        flags=re.I,
    )
    return text


# greedy decoding: generate the title one token at a time
# normalize user text -> tokenize -> encode w/ src_vocab -> run encoder once -> 
# start decoder with <sos> -> repeatedly do {feed the last predicted token into 
# decoder, take argmax of logits, append predicted token, and stop at <eos> or max_len}
# e.g. "i have chicken rice garlic" -> "i have chicken rice garlic" (as possible title output)
def greedy(model, src_vocab: Vocab, tgt_vocab: Vocab, text: str, device, max_len: int = 40) -> str:
    model.eval()
    ids = src_vocab.encode(tokenize(normalize_user_text(text)))
    if not ids:
        ids = [src_vocab.stoi[Vocab.UNK]]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    lens = torch.tensor([len(ids)], dtype=torch.long, device=device)
    sos = tgt_vocab.stoi[Vocab.SOS]
    eos = tgt_vocab.stoi[Vocab.EOS]
    with torch.no_grad():
        h = model.encoder(src, lens)
        out_ids = [sos]
        for _ in range(max_len):
            tgt = torch.tensor([[out_ids[-1]]], dtype=torch.long, device=device)
            logits, h = model.decoder(tgt, h)
            ni = int(logits[0, -1].argmax().item())
            out_ids.append(ni)
            if ni == eos:
                break
    return clean_generated_text(tgt_vocab.decode(out_ids))

# reconstructs everything from a saved training checkpoint for inference: 
# source vocab, target vocab, encoder, decoder, seq2seq model weights, and device
def load_checkpoint(ckpt_path: str, device=None):
    """Load model + vocabs from a training checkpoint."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab = Vocab.from_itos(ck["src_itos"])
    tgt_vocab = Vocab.from_itos(ck["tgt_itos"])
    enc = Encoder(
        len(src_vocab), ck["emb"], ck["hidden"],
        src_vocab.stoi[Vocab.PAD], num_layers=ck["layers"],
    )
    dec = Decoder(
        len(tgt_vocab), ck["emb"], ck["hidden"],
        tgt_vocab.stoi[Vocab.PAD], num_layers=ck["layers"],
    )
    model = Seq2Seq(enc, dec).to(device)
    model.load_state_dict(ck["model"])
    return model, src_vocab, tgt_vocab, device
