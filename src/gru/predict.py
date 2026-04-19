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


def normalize_user_text(s: str) -> str:
    """Normalize free-form user text into the prompt format seen during training."""
    s = (s or "").lower().strip()
    if not s.startswith("i have"):
        s = "i have " + re.sub(r"^i\s+have\s+", "", s, flags=re.I)
    return s


def clean_generated_text(text: str) -> str:
    """Remove special tokens and stray leading contractions from model output."""
    text = re.sub(r"\s+", " ", (text or "").replace(Vocab.UNK, " ")).strip()
    text = re.sub(
        r"^(?:['`’]?s|['`’]?d|['`’]?ll|['`’]?re|['`’]?ve|['`’]?m)\b[\s,.:;-]*",
        "",
        text,
        flags=re.I,
    )
    return text


def greedy(model, src_vocab: Vocab, tgt_vocab: Vocab, text: str, device, max_len: int = 40) -> str:
    """Greedily decode a title one token at a time until EOS or max length."""
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
