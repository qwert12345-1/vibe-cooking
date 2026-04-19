"""Jaccard, coverage, match-ratio, and missing-ingredient penalty metrics.

These are set-based scores computed on the *resolved* ingredient names — they
complement the TF-IDF cosine score, which is continuous and dominated by rare
ingredients. The recommender fuses both.
"""
from __future__ import annotations

import numpy as np

from .constants import PANTRY_STAPLES


def jaccard(user: set[str], recipe: set[str]) -> float:
    """|A ∩ B| / |A ∪ B|."""
    if not user and not recipe:
        return 0.0
    inter = user & recipe
    union = user | recipe
    return len(inter) / len(union) if union else 0.0


def coverage(user: set[str], recipe: set[str]) -> float:
    """Fraction of the recipe's ingredients that the user already has.

    Pantry staples are excluded from the denominator — almost every recipe
    contains salt/oil/water, and requiring the user to list them would be
    annoying.
    """
    core = {ing for ing in recipe if ing not in PANTRY_STAPLES}
    if not core:
        return 1.0
    return len(user & core) / len(core)


def weighted_coverage(user: set[str], recipe_weights: dict[str, float]) -> float:
    """Fraction of the recipe's **non-pantry gram mass** that the user has.

    `recipe_weights` maps canonical food name → grams. Pantry staples must
    already be stripped out by the caller.

    Counting ingredients equally treats 500g of chicken and 5g of basil
    as the same "one-of-three" check, which lets a caprese-style recipe
    (tomato + basil + bread) outscore a chicken-centric recipe for a user
    who typed {chicken, tomato, basil} — because both happen to hit 2/3
    of the items. Weighing by mass captures the intuition that chicken
    is the MAIN ingredient: a user who has chicken has a large share of
    a chicken recipe's mass, but only a trivial share of a bread recipe's.

    Falls back to the unweighted coverage when mass data is missing (all
    zero or empty) so the score still functions on older data.
    """
    total = sum(recipe_weights.values())
    if total <= 0 or not recipe_weights:
        return coverage(user, set(recipe_weights.keys()))
    user_mass = sum(w for ing, w in recipe_weights.items() if ing in user)
    return user_mass / total


def match_ratio(user: set[str], recipe: set[str]) -> float:
    """Fraction of the user's input that is actually used by the recipe.

    High match_ratio = "this recipe uses most of what I typed", which is
    different from coverage ("I have most of what this recipe needs").
    """
    if not user:
        return 0.0
    return len(user & recipe) / len(user)


def weighted_match_ratio(
    user: set[str],
    recipe: set[str],
    user_weights: dict[str, float],
) -> float:
    """IDF-weighted version of match_ratio.

    `user_weights` maps user-input tokens → importance weight (typically
    the vectorizer's IDF). Common ingredients like 'salt' get a weight of
    ~1, rare ones like 'saffron' get high weights. The ratio becomes:

        Σ w(i) for i in (user ∩ recipe)   /   Σ w(i) for i in user

    meaning: "what fraction of the INFORMATION I put into my query did
    this recipe honour?" A user who types 'chicken, saffron, salt' and
    gets a recipe with chicken+saffron but no salt covers nearly 100%
    of the information (salt carries almost none), whereas a recipe
    with just chicken+salt covers only the chicken share (saffron is
    the expensive ask). Missing tokens fall back to weight 1.0.
    """
    if not user:
        return 0.0
    overlap = user & recipe
    num = sum(user_weights.get(i, 1.0) for i in overlap)
    den = sum(user_weights.get(i, 1.0) for i in user)
    if den <= 0:
        return 0.0
    return num / den


def missing_ingredients(user: set[str], recipe: set[str]) -> list[str]:
    """Non-staple ingredients the recipe needs but the user didn't list."""
    return sorted([ing for ing in recipe if ing not in user and ing not in PANTRY_STAPLES])


def missing_penalty(user: set[str], recipe: set[str], alpha: float = 0.08) -> float:
    """Multiplicative penalty ∈ (0, 1] that shrinks as more non-staples are missing.

    penalty = exp(-alpha * |missing|).
    alpha=0.08 means ~5 missing ingredients → 0.67 penalty, ~10 → 0.45.
    """
    missing = missing_ingredients(user, recipe)
    return float(np.exp(-alpha * len(missing)))


def name_match_bonus(
    user: set[str],
    recipe: set[str],
    recipe_name: str | None,
) -> float:
    """Fraction of recipe-name-mentioned non-pantry ingredients that the user has,
    scaled down for trivially simple recipes.

    Captures the "main ingredient" signal — ingredients called out in the
    recipe's name ("Grilled Beer Can Chicken" → {beer, chicken}) are almost
    always the dish's defining components. This is the classical IR **title-
    match boost**: a query term that appears in a document's title is worth
    much more than one buried in the body.

    **Complexity scaling**: a recipe with a single non-pantry ingredient
    ("Pressure-Rendered Chicken Fat" → just {chicken}) would naively score
    name_bonus = 1.0 any time a chicken user came along. But that recipe
    isn't really a "chicken dish" in any meaningful sense — chicken IS the
    whole thing. We scale the bonus by `min(1, recipe_nonpantry_size / 3)`
    so single-ingredient recipes see only 1/3 of the bonus, and the signal
    activates in full only for properly composed (≥3 non-pantry) dishes.
    """
    if not recipe_name:
        return 0.0
    name_lower = recipe_name.lower()
    core = {ing for ing in recipe if ing not in PANTRY_STAPLES}
    mentioned = {ing for ing in core if ing in name_lower}
    if not mentioned:
        return 0.0
    raw = len(user & mentioned) / len(mentioned)
    complexity_scale = min(1.0, len(core) / 3.0)
    return raw * complexity_scale


def fused_score(
    cosine: float,
    user: set[str],
    recipe: set[str],
    *,
    recipe_weights: dict[str, float] | None = None,
    user_weights: dict[str, float] | None = None,
    recipe_name: str | None = None,
    w_cos: float = 0.50,
    w_f1: float = 0.30,
    w_name: float = 0.20,
    penalty_alpha: float = 0.03,
) -> float:
    """Fusion of three signals + missing-ingredient penalty.

        score = w_cos·cosine  +  w_f1·F1(match_ratio, coverage)
              + w_name·name_match_bonus
        score *= exp(-α · |non-pantry missing|)

    Each signal captures a different failure mode:

    1. **cosine** — continuous TF-IDF similarity (rare-ingredient matches
       count more; dense coverage of the query vocabulary helps).

    2. **F1 of (match_ratio, coverage)** — symmetric set-overlap quality.
       F1 punishes trivial recipes that accidentally cover one user input
       with 1-ingredient lists (coverage = 1.0 but match_ratio = tiny).
       When `recipe_weights` is supplied, coverage is mass-weighted.

    3. **name_match_bonus** — the "main ingredient" signal. Recipes whose
       title calls out user-held ingredients (e.g. "chicken" in "Beer Can
       Chicken" when the user has chicken) score materially higher than
       recipes that merely contain accompaniment-level matches. This is
       the single most important correction over a pure-overlap score for
       the "I have chicken, tomato, basil → I want a chicken dish" case.
    """
    if recipe_weights is not None:
        cov = weighted_coverage(user, recipe_weights)
    else:
        cov = coverage(user, recipe)
    if user_weights is not None:
        mr = weighted_match_ratio(user, recipe, user_weights)
    else:
        mr = match_ratio(user, recipe)
    denom = mr + cov
    f1 = (2.0 * mr * cov / denom) if denom > 0 else 0.0
    nb = name_match_bonus(user, recipe, recipe_name)
    base = w_cos * cosine + w_f1 * f1 + w_name * nb
    return base * missing_penalty(user, recipe, alpha=penalty_alpha)
