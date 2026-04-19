"""[HAND-IMPLEMENTED] TF-IDF vectorizer + cosine similarity (numpy/scipy only).

We deliberately do not use scikit-learn's TfidfVectorizer. This module constructs
sparse CSR matrices directly from the dataset's per-recipe ingredient lists,
computes IDF with the standard smoothed formula, L2-normalizes the resulting
vectors, and exposes cosine similarity as a single sparse matrix product.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class TfidfVectorizer:
    """Ingredient-token TF-IDF vectorizer.

    - Vocabulary is supplied externally (built from the training corpus).
    - Term frequency is *binary* per recipe (an ingredient is either present or not);
      this matches how ingredient lists work — having "onion" twice in a recipe is
      not a stronger signal than having it once.
    - IDF uses the smoothed form: log((N+1)/(df+1)) + 1.
    - Output vectors are L2-normalized so cosine similarity reduces to a dot product.
    """

    vocab: list[str]
    idf: np.ndarray | None = None

    def __post_init__(self):
        self.token_to_idx: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}

    # ---- fit -------------------------------------------------------------
    def fit(self, documents: list[list[str]]) -> "TfidfVectorizer":
        """Compute IDF from a list of token lists."""
        n_docs = len(documents)
        df = np.zeros(len(self.vocab), dtype=np.float64)
        for doc in documents:
            seen = set()
            for tok in doc:
                idx = self.token_to_idx.get(tok)
                if idx is not None and idx not in seen:
                    df[idx] += 1.0
                    seen.add(idx)
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        return self

    # ---- transform -------------------------------------------------------
    def transform(self, documents: list[list[str]]) -> sparse.csr_matrix:
        """Transform docs → L2-normalized sparse TF-IDF matrix (n_docs, vocab_size)."""
        if self.idf is None:
            raise RuntimeError("Call fit() before transform().")
        rows, cols, data = [], [], []
        for i, doc in enumerate(documents):
            seen: dict[int, float] = {}
            for tok in doc:
                idx = self.token_to_idx.get(tok)
                if idx is None:
                    continue
                seen[idx] = 1.0  # binary TF
            for j, tf in seen.items():
                w = tf * self.idf[j]
                if w > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(w)
        X = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(documents), len(self.vocab)),
            dtype=np.float64,
        )
        return _l2_normalize_rows(X)

    def fit_transform(self, documents: list[list[str]]) -> sparse.csr_matrix:
        self.fit(documents)
        return self.transform(documents)

    # ---- query-vector helper --------------------------------------------
    def transform_query(self, tokens: list[str], weights: dict[str, float] | None = None) -> sparse.csr_matrix:
        """Build a query vector (1, vocab_size). Optional per-token weights in [0,1]."""
        if self.idf is None:
            raise RuntimeError("Call fit() before transform_query().")
        cols, data = [], []
        seen: dict[int, float] = {}
        for tok in tokens:
            idx = self.token_to_idx.get(tok)
            if idx is None:
                continue
            w = weights.get(tok, 1.0) if weights else 1.0
            # Keep the max weight if a token appears multiple times with different weights.
            seen[idx] = max(seen.get(idx, 0.0), w)
        for j, tf in seen.items():
            val = tf * self.idf[j]
            if val > 0:
                cols.append(j)
                data.append(val)
        rows = [0] * len(cols)
        q = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(1, len(self.vocab)),
            dtype=np.float64,
        )
        return _l2_normalize_rows(q)


def cosine_similarity(query: sparse.csr_matrix, matrix: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity between a (1, V) query and an (N, V) matrix → (N,) dense array.

    Assumes both inputs are already L2-normalized (as produced by TfidfVectorizer
    above), so cosine = dot product.
    """
    sims = matrix.dot(query.T)
    return np.asarray(sims.todense()).ravel()


def _l2_normalize_rows(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-wise L2 normalization of a sparse CSR matrix. In-place on a copy."""
    X = X.copy().astype(np.float64)
    # Compute ||row||_2 efficiently on CSR data.
    sq = X.multiply(X)
    row_sums = np.asarray(sq.sum(axis=1)).ravel()
    norms = np.sqrt(row_sums)
    norms[norms == 0] = 1.0  # avoid div-by-zero; zero rows remain zero
    # Divide each row by its norm using diagonal scaling.
    inv = sparse.diags(1.0 / norms)
    return (inv @ X).tocsr()
