"""Multi-stage input normalization pipeline.

Stages (tried in order; first successful stage wins):

    1. clean       — lowercase, strip punctuation, collapse whitespace
    2. lemmatize   — NLTK WordNet lemmatization to handle plurals/morphology
    3. exact       — hit in the recipe-corpus vocabulary
    4. alias       — hand-curated synonym / translation dictionary
    5. superordinate — abstract terms ("poultry") expand to a set
    6. fuzzy       — RapidFuzz Levenshtein against the vocabulary (typo recovery)
    7. semantic    — sentence-transformer cosine similarity (last-resort fallback)

Each token returns a `NormalizedToken` with the resolved canonical forms,
a confidence score, and the stage that resolved it — so the UI can ask the
user to confirm low-confidence matches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .constants import ALIASES, SUPERORDINATES


# dataclass to store the result of noramlizing one input token
@dataclass
class NormalizedToken:
    original: str
    canonical: list[str]      # resolved canonical name(s) — usually length 1
    stage: str                # which stage resolved this token
    confidence: float         # in [0, 1]; 1.0 means "certain"
    candidates: list[tuple[str, float]] = field(default_factory=list)  # alternatives for UI

    @property
    def needs_confirmation(self) -> bool:
        return self.confidence < 0.85 and self.stage in ("fuzzy", "semantic") # return true if the token was resolved using fuzzy or semantic stage

# stateful normalizer; construct once per app session and reuse
class Normalizer:

    # initialize the Normalizer with the recipe vocabulary and load optional AI models (SBERT/NLTK)
    def __init__(
        self,
        vocabulary: list[str],
        *,
        use_semantic: bool = True,
        fuzzy_threshold: float = 80.0,   # RapidFuzz partial_ratio, 0–100
        semantic_threshold: float = 0.55,
    ):
        self.vocabulary = vocabulary
        self.vocab_set = set(vocabulary)
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold

        self._lemmatizer = self._load_lemmatizer()
        self._fuzzy = self._load_fuzzy()
        if use_semantic:
            print("[normalizer] Loading SBERT model (first run downloads ~80MB)...",
                  flush=True)
            self._semantic = self._load_semantic()
        else:
            self._semantic = None
        self._vocab_embeddings = None
        if self._semantic is not None:
            print(f"[normalizer] Encoding {len(self.vocabulary)} vocab tokens with SBERT...",
                  flush=True)
            self._vocab_embeddings = self._semantic.encode(
                self.vocabulary, show_progress_bar=False, normalize_embeddings=True
            )
            print("[normalizer] SBERT ready.", flush=True)

    # ---- loaders (lazy, optional) ---------------------------------------
    @staticmethod
    def _load_lemmatizer():
        """Try to return a WordNetLemmatizer. If the corpus isn't installed
        and can't be downloaded (e.g. network failure), return None so the
        normalizer can fall back to a naive s-stripping heuristic. Never raise,
        never block on a network timeout."""
        import os
        if os.environ.get("RR_NO_NLTK", "0") == "1":
            return None
        try:
            import nltk  # noqa: F401
            from nltk.stem import WordNetLemmatizer
        except ImportError:
            return None
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            # Try a one-shot download with a short socket timeout so we don't
            # hang for minutes on a blocked network.
            import socket
            prev_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(5)
                ok = nltk.download("wordnet", quiet=True, raise_on_error=False)
            except Exception:
                ok = False
            finally:
                socket.setdefaulttimeout(prev_timeout)
            if not ok:
                print("[normalizer] NLTK wordnet not available — "
                      "falling back to naive singular/plural stripping. "
                      "Set RR_NO_NLTK=1 to silence this, or download the corpus "
                      "manually (see README).", flush=True)
                return None
            # Verify it actually landed on disk.
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                return None
        return WordNetLemmatizer()

    @staticmethod
    # loads the RapidFuzz library for high-speed typo correction (Levenshtein distance)
    def _load_fuzzy():
        try:
            from rapidfuzz import process, fuzz
            return (process, fuzz)
        except ImportError:
            return None

    @staticmethod
    # loads the SBERT neural network to perform mathematical semantic guesses
    def _load_semantic():
        from .embeddings import get_sbert_model
        return get_sbert_model()

    # ---- public API ------------------------------------------------------
    def normalize(self, raw: str) -> NormalizedToken:
        """Run one raw token through the pipeline."""
        cleaned = self._clean(raw)
        if not cleaned:
            return NormalizedToken(raw, [], "empty", 0.0)

        # Stage 3: exact hit on cleaned form
        if cleaned in self.vocab_set:
            return NormalizedToken(raw, [cleaned], "exact", 1.0)

        # Stage 2 + 3: lemmatize then retry exact
        lemma = self._lemmatize(cleaned)
        if lemma in self.vocab_set:
            return NormalizedToken(raw, [lemma], "lemmatize", 1.0)

        # Stage 4: alias dictionary
        if cleaned in ALIASES:
            target = ALIASES[cleaned]
            if target in self.vocab_set:
                return NormalizedToken(raw, [target], "alias", 1.0)
        if lemma in ALIASES:
            target = ALIASES[lemma]
            if target in self.vocab_set:
                return NormalizedToken(raw, [target], "alias", 1.0)

        # Stage 5: superordinate expansion
        if cleaned in SUPERORDINATES:
            expansion = [t for t in SUPERORDINATES[cleaned] if t in self.vocab_set]
            if expansion:
                return NormalizedToken(raw, expansion, "superordinate", 0.9)

        # Stage 6: fuzzy match against vocabulary
        fuzzy_match = self._fuzzy_lookup(cleaned)
        if fuzzy_match is not None:
            top, score, alts = fuzzy_match
            confidence = min(1.0, score / 100.0)
            return NormalizedToken(raw, [top], "fuzzy", confidence, alts)

        # Stage 7: semantic fallback
        semantic_match = self._semantic_lookup(cleaned)
        if semantic_match is not None:
            top, score, alts = semantic_match
            return NormalizedToken(raw, [top], "semantic", float(score), alts)

        return NormalizedToken(raw, [], "unresolved", 0.0)

    # convenience function to take a list of raw words and run `normalize` on all of them
    def normalize_batch(self, raw_inputs: list[str]) -> list[NormalizedToken]:
        return [self.normalize(r) for r in raw_inputs]

    # ---- stages ----------------------------------------------------------
    @staticmethod
    # violently strips out casing, weird punctuation, and collapses spaces
    def _clean(text: str) -> str:
        if not text:
            return ""
        s = text.lower().strip()
        s = re.sub(r"[^\w\s\u4e00-\u9fff-]", " ", s)  # keep ASCII word chars, CJK, hyphen
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # converts plural words to singular using linguistic models (chickens -> chicken)
    def _lemmatize(self, text: str) -> str:
        if self._lemmatizer is None:
            # Naive s-stripping fallback
            return text[:-1] if text.endswith("s") and len(text) > 3 else text
        parts = [self._lemmatizer.lemmatize(p, pos="n") for p in text.split()]
        return " ".join(parts)

    # attempts to fix spelling mistakes by finding the closest string in the vocabulary
    def _fuzzy_lookup(self, text: str) -> Optional[tuple[str, float, list[tuple[str, float]]]]:
        if self._fuzzy is None or not self.vocabulary:
            return None
        process, fuzz = self._fuzzy
        candidates = process.extract(text, self.vocabulary, scorer=fuzz.WRatio, limit=5)
        if not candidates:
            return None
        top, top_score, _ = candidates[0]
        if top_score < self.fuzzy_threshold:
            return None
        alts = [(c[0], c[1] / 100.0) for c in candidates]
        return top, top_score, alts

    # last resort: translates word to a 384-dimension vector, and finds the vocabulary word with the highest Cosine Similarity!
    def _semantic_lookup(self, text: str) -> Optional[tuple[str, float, list[tuple[str, float]]]]:
        if self._semantic is None or self._vocab_embeddings is None:
            return None
        import numpy as np
        q = self._semantic.encode([text], normalize_embeddings=True)[0]
        sims = self._vocab_embeddings @ q
        top_idx = int(sims.argmax())
        top_score = float(sims[top_idx])
        if top_score < self.semantic_threshold:
            return None
        # Top-5 alternatives
        top_k = np.argsort(-sims)[:5]
        alts = [(self.vocabulary[int(i)], float(sims[int(i)])) for i in top_k]
        return self.vocabulary[top_idx], top_score, alts


# helper function flattens a nested structure of NormalizedToken objects into a flat list of strings
def flatten_resolved(tokens: list[NormalizedToken]) -> list[str]:
    # initialize an empty list to store the canonical forms and a set to keep track of seen canonical forms
    out: list[str] = []
    seen: set[str] = set()

    # iterate through the tokens and add the canonical forms to the output list
    for t in tokens:
        for c in t.canonical:
            if c not in seen:
                out.append(c)
                seen.add(c)
    return out
