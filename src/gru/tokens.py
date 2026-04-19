"""Lightweight tokenizer + ingredient canonicalizer used by the GRU pipeline.

Extracted from the original vibe-cooking-main project's `data.py` and
`allrecipes_data.py` so this sub-package can stand on its own without
dragging in the datasets / Hugging Face dependency chain.
"""
from __future__ import annotations

import re

_PARENS_RE = re.compile(r"\([^)]*\)")
_LEADING_QTY_RE = re.compile(
    r"^\s*(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*"
)
_LEADING_MULT_RE = re.compile(r"^\s*\d+\s*(?:x|X)\s+")
_NON_WORD_RE = re.compile(r"[^a-z\s]")

_UNIT_WORDS = {
    "c", "cup", "cups", "t", "tbsp", "tablespoon", "tablespoons",
    "tsp", "teaspoon", "teaspoons", "oz", "ounce", "ounces",
    "lb", "lbs", "pound", "pounds", "can", "cans", "jar", "jars",
    "pkg", "pkgs", "package", "packages", "box", "boxes", "bag", "bags",
    "bottle", "bottles", "quart", "quarts", "pint", "pints",
    "gallon", "gallons", "ml", "l", "liter", "liters",
    "clove", "cloves", "slice", "slices", "dash", "pinch", "stick", "sticks",
}

_DESCRIPTOR_WORDS = {
    "about", "approximately", "boned", "broken", "chopped", "cold", "cooked",
    "crushed", "cubed", "diced", "divided", "drained", "dry", "extra",
    "extra-virgin", "finely", "firmly", "fresh", "frozen", "grated", "ground",
    "halved", "large", "lean", "light", "lite", "melted", "medium", "minced",
    "optional", "packed", "peeled", "powdered", "prepared", "quartered", "ripe",
    "room", "seeded", "shredded", "sifted", "small", "soft", "softened",
    "thinly", "toasted", "warm", "white", "yellow",
}

_CONNECTOR_WORDS = {"and", "or", "of", "for", "with", "plus", "to", "taste"}


def tokenize(s: str) -> list[str]:
    """Lowercase and tokenize simple recipe text into word pieces."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return [t for t in s.split() if t]


def normalize_ingredient_name(line: str) -> str:
    """Best-effort canonical ingredient phrase for matching and display.

    Copied verbatim from the original project so the RAG composer sees the
    same canonicalization it was designed for.
    """
    s = (line or "").lower().strip()
    if not s:
        return ""
    s = re.split(r"\s+or\s+", s, maxsplit=1)[0]
    s = _PARENS_RE.sub(" ", s)
    s = s.replace("-", " ")
    s = re.split(r"\s*,\s*", s, maxsplit=1)[0]
    s = _LEADING_MULT_RE.sub("", s)

    while True:
        updated = _LEADING_QTY_RE.sub("", s).strip()
        if updated == s:
            break
        s = updated

    tokens: list[str] = []
    for tok in s.split():
        tok = _NON_WORD_RE.sub("", tok)
        if not tok:
            continue
        if tok in _UNIT_WORDS or tok in _DESCRIPTOR_WORDS or tok in _CONNECTOR_WORDS:
            continue
        if tok.isdigit():
            continue
        tokens.append(tok)

    if not tokens:
        return ""
    if len(tokens) > 3:
        tokens = tokens[-3:]
    return " ".join(tokens).strip()
