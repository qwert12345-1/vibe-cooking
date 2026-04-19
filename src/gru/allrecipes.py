"""Local AllRecipes parser + retriever for GRU grounding.

This keeps the GRU branch tied to the same `corbt/all-recipes` corpus used in
the sibling demo project, while staying self-contained inside this repo.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
from typing import Optional

from .tokens import tokenize

_SECTION_RE = re.compile(r"^\s*(Ingredients|Directions)\s*:\s*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-*]\s*")
_WS_RE = re.compile(r"\s+")
_STEP_END_RE = re.compile(r"[.!?][\"')\]]?\s*$")

DEFAULT_DATASET = os.environ.get("RR_GRU_ALLRECIPES_DATASET", "corbt/all-recipes")
DEFAULT_LIMIT = int(os.environ.get("RR_GRU_ALLRECIPES_LIMIT", "50000"))


@dataclass
class ParsedRecipe:
    title: str
    ingredient_lines: list[str]
    directions: list[str]
    raw_text: str


_recipes_cache: list[ParsedRecipe] | None = None
_cache_limit: int | None = None
_load_error: Optional[str] = None
_cache_lock = threading.Lock()


def _clean_line(line: str) -> str:
    line = _BULLET_RE.sub("", (line or "").strip())
    line = _WS_RE.sub(" ", line)
    return line.strip()


def _normalize_text(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_clean_line(line) for line in text.split("\n")]
    return [line for line in lines if line]


def _merge_direction_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    buf: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        candidate = " ".join(buf).strip()
        if _STEP_END_RE.search(candidate):
            merged.append(candidate)
            buf = []
    if buf:
        merged.append(" ".join(buf).strip())
    return merged


def parse_recipe_blob(text: str) -> ParsedRecipe | None:
    """Parse one raw `corbt/all-recipes` text blob into structured fields."""
    lines = _normalize_text(text)
    if not lines:
        return None

    title = lines[0]
    current = None
    ingredients: list[str] = []
    directions: list[str] = []

    for line in lines[1:]:
        match = _SECTION_RE.match(line)
        if match:
            current = match.group(1).lower()
            continue
        if current == "ingredients":
            ingredients.append(line)
        elif current == "directions":
            directions.append(line)

    directions = _merge_direction_lines(directions)
    if not ingredients and not directions:
        return None

    return ParsedRecipe(
        title=title,
        ingredient_lines=ingredients,
        directions=directions,
        raw_text=text,
    )


def _local_arrow_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    return sorted(
        root.glob(
            "models/datasets/corbt___all-recipes/default/0.0.0/*/all-recipes-train-*.arrow"
        )
    )


def _load_raw_rows(limit: int = 0):
    """Yield rows from the local cache when available, else from HF datasets."""
    local_paths = _local_arrow_paths()

    if local_paths:
        from datasets import Dataset

        yielded = 0
        for path in local_paths:
            shard = Dataset.from_file(str(path))
            for row in shard:
                yield row
                yielded += 1
                if limit and yielded >= limit:
                    return
        return

    from datasets import load_dataset

    ds = load_dataset(DEFAULT_DATASET, split="train")
    total = len(ds)
    upper = min(limit, total) if limit else total
    for idx in range(upper):
        yield ds[idx]


def load_allrecipes(limit: int = DEFAULT_LIMIT) -> list[ParsedRecipe]:
    """Load and parse the local AllRecipes corpus used for GRU grounding."""
    global _recipes_cache, _cache_limit, _load_error

    requested = max(0, int(limit))
    if _recipes_cache is not None:
        if requested == 0 and _cache_limit == 0:
            return _recipes_cache
        if requested and _cache_limit is not None and (_cache_limit == 0 or _cache_limit >= requested):
            return _recipes_cache[:requested]

    with _cache_lock:
        if _recipes_cache is not None:
            if requested == 0 and _cache_limit == 0:
                return _recipes_cache
            if requested and _cache_limit is not None and (_cache_limit == 0 or _cache_limit >= requested):
                return _recipes_cache[:requested]
        if _load_error is not None:
            raise RuntimeError(_load_error)

        parsed: list[ParsedRecipe] = []
        try:
            for row in _load_raw_rows(limit=requested):
                recipe = parse_recipe_blob(str(row.get("input", "")))
                if recipe is not None:
                    parsed.append(recipe)
        except Exception as exc:  # pragma: no cover - defensive
            _load_error = f"Failed to load AllRecipes grounding corpus: {exc}"
            raise RuntimeError(_load_error) from exc

        _recipes_cache = parsed
        _cache_limit = requested
        return parsed


def recipe_to_retrieval_text(recipe: ParsedRecipe) -> str:
    return " | ".join(
        [
            recipe.title.lower(),
            " ".join(recipe.ingredient_lines).lower(),
            " ".join(recipe.directions).lower(),
        ]
    )


def _ingredient_query_text(user_ingredients: str | list[str]) -> str:
    if isinstance(user_ingredients, str):
        return user_ingredients
    return ", ".join(str(item).strip() for item in user_ingredients if str(item).strip())


def score_recipe(recipe: ParsedRecipe, user_ingredients: str | list[str]) -> float:
    """Rank recipes by ingredient overlap with a small title bonus."""
    query_tokens = set(tokenize(_ingredient_query_text(user_ingredients)))
    doc_tokens = set(tokenize(recipe_to_retrieval_text(recipe)))
    if not query_tokens:
        return 0.0

    overlap = len(query_tokens & doc_tokens) / len(query_tokens)

    title_bonus = 0.0
    title_tokens = set(tokenize(recipe.title))
    if query_tokens & title_tokens:
        title_bonus = 0.10

    return overlap + title_bonus


def retrieve_recipes(
    recipes: list[ParsedRecipe],
    user_ingredients: str | list[str],
    *,
    top_k: int = 5,
) -> list[tuple[float, ParsedRecipe]]:
    scored: list[tuple[float, ParsedRecipe]] = []
    for recipe in recipes:
        score = score_recipe(recipe, user_ingredients)
        if score > 0:
            scored.append((score, recipe))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def build_matches(
    ingredients: list[str],
    *,
    predicted_title: str | None = None,
    top_k: int = 5,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """Return RAG-ready matches from the AllRecipes retrieval corpus."""
    recipes = load_allrecipes(limit=limit)
    query = ", ".join(ingredients)
    if predicted_title:
        query = f"{query}, {predicted_title}"

    matches: list[dict] = []
    for score, recipe in retrieve_recipes(recipes, query, top_k=top_k):
        matches.append(
            {
                "score": round(float(score), 3),
                "title": recipe.title,
                "ingredient_lines": recipe.ingredient_lines,
                "directions": recipe.directions,
            }
        )
    return matches


def check_allrecipes() -> tuple[bool, str]:
    """Lightweight health check for the AllRecipes grounding corpus."""
    try:
        local_paths = _local_arrow_paths()
        if local_paths:
            return True, f"Ready via {len(local_paths)} local shard(s)"
        load_allrecipes(limit=1)
        return True, f"Ready via {DEFAULT_DATASET}"
    except Exception as exc:
        return False, str(exc)
