"""SEQ2SEQ-based creative recipe generator.

Bridges the separately-trained SEQ2SEQ seq2seq title model + RAG composer from
the sibling demo project into this app. The flow is:

    1. User ingredients → normalize_user_text()
    2. SEQ2SEQ seq2seq → predicted recipe title ("Chicken Something")
    3. Retrieve top-K real recipes from the local `corbt/all-recipes` corpus
       used by the original demo, including ingredient lines + directions
    4. Feed (title, retrieval matches) to rag.generate_recipe_draft(), which
       stitches an ingredient list + step-by-step instructions grounded in
       those retrieval matches
    5. Render as markdown suitable for the UI

This sits alongside `src/creative.py` (the LLM generator) so the UI can
offer both as interchangeable "creative recipe" backends.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CKPT = os.environ.get(
    "RR_SEQ2SEQ_CKPT",
    str(Path(__file__).resolve().parent.parent / "seq2seq_checkpoints" / "best.pt"),
)
DISABLED = os.environ.get("RR_NO_SEQ2SEQ", "0") == "1"


@dataclass
class SEQ2SEQRecipeResult:
    ok: bool
    text: str = ""                  # rendered markdown on success, error message on failure
    title: str = ""                 # SEQ2SEQ-predicted title
    model: str = "seq2seq-seq2seq"


# --------------------------------------------------------------------------
# Lazy-load the seq2seq bundle (model + vocabs + device).
# --------------------------------------------------------------------------
_bundle = None
_bundle_lock = threading.Lock()
_load_error: Optional[str] = None


def _load_bundle():
    global _bundle, _load_error
    if DISABLED:
        _load_error = "SEQ2SEQ disabled via RR_NO_SEQ2SEQ=1."
        return None
    if _bundle is not None:
        return _bundle
    if _load_error is not None:
        return None
    with _bundle_lock:
        if _bundle is not None:
            return _bundle
        if _load_error is not None:
            return None
        ckpt = Path(DEFAULT_CKPT)
        if not ckpt.exists():
            _load_error = f"SEQ2SEQ checkpoint not found at {ckpt}."
            return None
        try:
            # Deferred import so starting the app without torch doesn't crash.
            from .seq2seq.predict import load_checkpoint
            print(f"[seq2seq] Loading checkpoint {ckpt} ...", flush=True)
            _bundle = load_checkpoint(str(ckpt))
            _, _, _, dev = _bundle
            print(f"[seq2seq] Loaded on {dev}.", flush=True)
        except Exception as e:  # pragma: no cover — defensive
            _load_error = f"Failed to load SEQ2SEQ: {e}"
            print(f"[seq2seq] {_load_error}", flush=True)
            return None
    return _bundle


def check_seq2seq() -> tuple[bool, str]:
    """Non-blocking health check for the UI."""
    if DISABLED:
        return False, "SEQ2SEQ disabled (RR_NO_SEQ2SEQ=1)"
    ckpt = Path(DEFAULT_CKPT)
    if not ckpt.exists():
        return False, f"SEQ2SEQ checkpoint missing at {ckpt}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed"
    try:
        from .seq2seq.allrecipes import check_allrecipes

        ok, detail = check_allrecipes()
        if not ok:
            return False, f"AllRecipes grounding unavailable: {detail}"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"AllRecipes grounding unavailable: {e}"
    return True, f"Ready · checkpoint cached at {ckpt.name}"


# --------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------
def _render_markdown(draft: dict) -> str:
    import re

    def _clean_line(raw: object) -> str:
        """Collapse any internal whitespace (newlines, tabs, leading spaces)
        to a single space and strip ends. Required because RAG-composed lines
        sometimes come in with embedded newlines or leading whitespace, which
        markdown interprets as a nested sub-list and renders a stray empty
        bullet followed by an indented child."""
        return re.sub(r"\s+", " ", str(raw or "")).strip()

    title = _clean_line(draft.get("title")) or "SEQ2SEQ Recipe"
    ings = [_clean_line(x) for x in (draft.get("ingredient_lines") or [])]
    ings = [x for x in ings if x]
    steps = [_clean_line(x) for x in (draft.get("directions") or [])]
    steps = [x for x in steps if x]
    notes = [_clean_line(x) for x in (draft.get("notes") or [])]
    notes = [x for x in notes if x]

    parts: list[str] = []
    parts.append(f"# {title}\n")
    if ings:
        parts.append("## Ingredients")
        parts.append("\n".join(f"- {line}" for line in ings))
        parts.append("")
    if steps:
        parts.append("## Steps")
        parts.append("\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps)))
        parts.append("")
    if notes:
        parts.append("## Notes")
        parts.append("\n".join(f"- {n}" for n in notes))
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _build_matches_from_allrecipes(
    ingredients: list[str],
    *,
    predicted_title: str,
) -> list[dict]:
    """Pull top-K real recipes from the local AllRecipes corpus."""
    try:
        from .seq2seq.allrecipes import build_matches

        return build_matches(ingredients, predicted_title=predicted_title, top_k=5)
    except Exception as e:
        print(f"[seq2seq] allrecipes grounding failed ({e})")
        return []


def generate_seq2seq_recipe(
    ingredients: list[str],
    *,
    engine,
    pantry_items: Optional[list[str]] = None,
) -> SEQ2SEQRecipeResult:
    """Given a list of ingredient strings, return a SEQ2SEQ-drafted recipe.

    `engine` is accepted for backward compatibility with the UI but the SEQ2SEQ
    path now grounds against local AllRecipes matches instead of the main
    recommender corpus. `pantry_items` is an optional list of pantry staples
    the user has on hand.
    """
    if not ingredients:
        return SEQ2SEQRecipeResult(ok=False, text="No ingredients — type something first.")

    bundle = _load_bundle()
    if bundle is None:
        return SEQ2SEQRecipeResult(ok=False, text=f"⚠️ SEQ2SEQ unavailable: {_load_error}")

    from .seq2seq.predict import greedy
    from .seq2seq.rag import generate_recipe_draft

    model, src_vocab, tgt_vocab, device = bundle
    user_text = ", ".join(ingredients)

    # 1) Predict the title with the trained seq2seq.
    try:
        predicted_title = greedy(model, src_vocab, tgt_vocab, user_text, device)
    except Exception as e:
        return SEQ2SEQRecipeResult(ok=False, text=f"⚠️ SEQ2SEQ inference failed: {e}")

    # 2) Pull retrieval-grounded examples from the local AllRecipes corpus.
    matches = _build_matches_from_allrecipes(ingredients, predicted_title=predicted_title)

    # 3) Compose the full recipe with the RAG heuristic.
    try:
        draft = generate_recipe_draft(
            user_text, pantry_items or [], matches,
            predicted_title=predicted_title,
        )
    except Exception as e:
        return SEQ2SEQRecipeResult(
            ok=False, text=f"⚠️ SEQ2SEQ draft composition failed: {e}",
            title=predicted_title,
        )

    md = _render_markdown(draft)
    return SEQ2SEQRecipeResult(ok=True, text=md, title=str(draft.get("title") or predicted_title))
