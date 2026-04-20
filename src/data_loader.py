"""This downloads recipes from Hugging Face, parses messy JSON field, cleans ingredient
names, and converts everything into a structured Recipe object. It also builds and caches
the dataset so we don't have to reload it everytime we run it. """
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CACHE_DIR = Path("cache")
DATA_DIR = Path("data")
DATASET_ID = "datahiveai/recipes-with-nutrition"


# dataclass to store each recipe, giving the rest of the code a clean object
# to work with instead of messy raw dataset rows
@dataclass
class Recipe:
    id: int
    name: str
    url: str
    image: str
    ingredients: list[str]           # canonical, normalized food names
    ingredients_raw: list[dict]      # original {food, weight, measure} dicts
    cuisine: list[str]
    diet_labels: list[str]
    health_labels: list[str]
    meal_type: list[str]
    calories: float
    total_time: float
    servings: float
    extras: dict = field(default_factory=dict)


# creates the cache and data dictionaries if not already existing
def _ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# columns often store list/dictionary fields as JSON strings — parse defensively
def _parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # Try single-quote → double-quote fallback (common dirty-CSV pattern)
            try:
                return json.loads(s.replace("'", '"'))
            except json.JSONDecodeError:
                return s
    return value


# normalizes an ingredient string by lwer casing it, trimming whitespace, collapse
# repeated spaces, and stripping punctuations from the ends (cleanup step before storing ingredients)
def _clean_food_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:()[]{}\"'")
    return s


# parses the raw ingredient field and return a list of clean ingredient names and 
# a list of the original ingredient dicts (giving both a canonical ingredient and original metadata)
def _extract_ingredients(raw_field: Any) -> tuple[list[str], list[dict]]:
    parsed = _parse_json_field(raw_field)
    if not isinstance(parsed, list):
        return [], []
    names, dicts = [], []
    for item in parsed:
        if isinstance(item, dict):
            food = _clean_food_name(item.get("food", ""))
            if food:
                names.append(food)
                dicts.append(item)
        elif isinstance(item, str):
            food = _clean_food_name(item)
            if food:
                names.append(food)
                dicts.append({"food": food})
    return names, dicts


# helper to convert a value into a list of strings (used for cuisine, diet_labels, health_labels, meal_type
# since they may come in as JSON strings, single strings, or list already)
def _as_list(value: Any) -> list[str]:
    parsed = _parse_json_field(value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    if isinstance(parsed, str):
        return [parsed] if parsed else []
    return []


# helper to convert a value into a float (used for calories, total_time, servings) and return default
# value if the value is None or empty string or cannot be converted to float
def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# main dataset-loading function: first checks whether a cache file already exists, if so, it loads the cached data
# otherwise, it loads the dataset from Hugging Face and caches it
def load_recipes(use_cache: bool = True, limit: int | None = None) -> list[Recipe]:
    _ensure_dirs()
    cache_path = CACHE_DIR / "recipes.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            recipes = pickle.load(f)
        if limit:
            return recipes[:limit]
        return recipes

    from datasets import load_dataset  # lazy import

    ds = load_dataset(DATASET_ID, split="train", cache_dir=str(DATA_DIR))
    recipes: list[Recipe] = []
    for idx, row in enumerate(ds):
        names, dicts = _extract_ingredients(row.get("ingredients"))
        if not names:
            continue
        # The dataset uses snake_case columns: recipe_name, image_url, cuisine_type, etc.
        # Fall back across a few variants so this loader works across dataset revisions.
        recipe = Recipe(
            id=idx,
            name=str(
                row.get("recipe_name")
                or row.get("label")
                or row.get("name")
                or row.get("title")
                or f"recipe-{idx}"
            ).strip(),
            url=str(row.get("url") or "").strip(),
            image=str(row.get("image_url") or row.get("image") or "").strip(),
            ingredients=names,
            ingredients_raw=dicts,
            cuisine=[c.lower() for c in _as_list(row.get("cuisine_type") or row.get("cuisineType"))],
            diet_labels=_as_list(row.get("diet_labels") or row.get("dietLabels")),
            health_labels=_as_list(row.get("health_labels") or row.get("healthLabels")),
            meal_type=[m.lower() for m in _as_list(row.get("meal_type") or row.get("mealType"))],
            calories=_as_float(row.get("calories")),
            total_time=_as_float(row.get("total_time") or row.get("totalTime")),
            servings=_as_float(row.get("servings") or row.get("yield"), default=1.0),
        )
        recipes.append(recipe)

    # cache the loaded recipes
    with open(cache_path, "wb") as f:
        pickle.dump(recipes, f)
    if limit:
        return recipes[:limit]
    return recipes


# computes the ingredient vocabulary by counting in how many recipes each ingredient appears
# and returns a sorted list of ingredients that appear in at least "min_df" recipes 
# (avoided clutter from extremely rare tokens)
def build_vocabulary(recipes: list[Recipe], min_df: int = 3) -> list[str]:
    from collections import Counter
    df = Counter()
    for r in recipes:
        for ing in set(r.ingredients):
            df[ing] += 1
    vocab = sorted([t for t, c in df.items() if c >= min_df])
    return vocab


# saving the covabulary to a pickle file (easy to be reused later)
def save_vocab(vocab: list[str], path: str | Path = CACHE_DIR / "vocab.pkl") -> None:
    _ensure_dirs()
    with open(path, "wb") as f:
        pickle.dump(vocab, f)


# loads a previously saved vocab from disk if exists, if not, return None
def load_vocab(path: str | Path = CACHE_DIR / "vocab.pkl") -> list[str] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
