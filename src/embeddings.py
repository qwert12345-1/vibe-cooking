"""Shared SBERT loader — keep a single model instance across the app.

Both the Normalizer (for semantic-fallback token resolution) and the
Recommender (for dense recipe retrieval + semantic cluster labels) need the
same embedding model. Loading it twice would double the memory footprint and
startup time, so this module owns the one shared copy.
"""
from __future__ import annotations

import os
# HF mirror for restricted networks — set before importing any huggingface lib.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


_MODEL = None
_MODEL_NAME: str | None = None


def get_sbert_model(model_name: str = DEFAULT_SBERT_MODEL):
    """Return the shared SBERT model (loading it lazily on first call).

    Tries the local cache first (`local_files_only=True`) so a restricted
    network can't block startup with a HEAD timeout against huggingface.co.
    Falls back to an online fetch only if the cache is genuinely empty.
    """
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    # 1) offline-first attempt using the project-local cache
    try:
        _MODEL = SentenceTransformer(model_name, local_files_only=True)
        _MODEL_NAME = model_name
        return _MODEL
    except Exception:
        pass
    # 2) online fallback
    try:
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
        return _MODEL
    except Exception as e:  # network failures, etc.
        print(f"[embeddings] Failed to load SBERT model {model_name!r}: {e}")
        return None


def encoded_dim(model_name: str = DEFAULT_SBERT_MODEL) -> int:
    """384 for the default MiniLM; fallback to querying the model."""
    if model_name == DEFAULT_SBERT_MODEL:
        return 384
    m = get_sbert_model(model_name)
    if m is None:
        return 0
    return int(m.get_sentence_embedding_dimension())
