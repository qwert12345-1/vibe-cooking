"""End-to-end smoke test over a tiny synthetic corpus."""
import numpy as np
import pytest

from src.data_loader import Recipe, build_vocabulary
from src.kmeans import fit_kmeans
from src.normalizer import Normalizer
from src.recommender import Filters, RecipeRecommender
from src.tfidf import TfidfVectorizer


def _toy_recipes() -> list[Recipe]:
    def make(idx, name, ings, cuisine, diet, cal):
        return Recipe(
            id=idx, name=name, url="", image="",
            ingredients=ings, ingredients_raw=[{"food": i} for i in ings],
            cuisine=cuisine, diet_labels=diet, health_labels=diet,
            meal_type=["dinner"], calories=cal, total_time=30, servings=2,
        )
    return [
        make(0, "Chicken Tomato Stew",
             ["chicken", "tomato", "onion", "garlic", "salt"], ["italian"], ["High-Protein"], 420),
        make(1, "Beef and Carrot",
             ["beef", "carrot", "onion", "potato", "salt"], ["british"], [], 650),
        make(2, "Chicken Fried Rice",
             ["chicken", "rice", "soy sauce", "ginger", "onion", "egg"], ["chinese"], [], 550),
        make(3, "Mapo Tofu",
             ["tofu", "soy sauce", "ginger", "chili pepper", "green onion"], ["chinese"], ["Vegetarian"], 380),
        make(4, "Chocolate Brownie",
             ["flour", "sugar", "butter", "egg", "cocoa"], ["american"], ["Vegetarian"], 320),
        make(5, "Vegan Fried Rice",
             ["rice", "soy sauce", "ginger", "carrot", "tofu", "green onion"], ["chinese"], ["Vegan"], 400),
    ]


@pytest.fixture(scope="module")
def engine():
    recipes = _toy_recipes()
    vocab = build_vocabulary(recipes, min_df=1)
    vec = TfidfVectorizer(vocab).fit([r.ingredients for r in recipes])
    matrix = vec.transform([r.ingredients for r in recipes])
    # Tiny 2-D projection via numpy SVD (no need for full engine build).
    dense = matrix.toarray().astype(np.float64)
    U, S, _ = np.linalg.svd(dense, full_matrices=False)
    proj = (U * S)[:, :2]
    km = fit_kmeans(proj, k=3, n_init=5, seed=0)
    norm = Normalizer(vocab, use_semantic=False)
    return RecipeRecommender(recipes, vocab, vec, matrix, km.labels, proj, norm)


def test_chicken_query_ranks_chicken_recipes(engine):
    ranked = engine.recommend(["chicken", "onion", "tomato"], top_k=3)
    assert ranked
    assert "chicken" in ranked[0].recipe.name.lower()


def test_excluded_ingredient_filter(engine):
    ranked = engine.recommend(
        ["chicken", "onion"], top_k=5, filters=Filters(excluded_ingredients=["rice"])
    )
    assert all("rice" not in r.recipe.ingredients for r in ranked)


def test_diet_filter_vegan(engine):
    ranked = engine.recommend(
        ["rice", "soy sauce", "ginger"], top_k=5, filters=Filters(diet=["Vegan"])
    )
    assert ranked
    for r in ranked:
        labels = {d.lower() for d in (r.recipe.diet_labels + r.recipe.health_labels)}
        assert "vegan" in labels


def test_cuisine_filter_chinese(engine):
    ranked = engine.recommend(
        ["chicken", "rice"], top_k=5, filters=Filters(cuisine=["chinese"])
    )
    assert ranked
    for r in ranked:
        assert "chinese" in r.recipe.cuisine


def test_chinese_input_translated(engine):
    # "洋葱" = onion; normalizer should translate, and result should be non-empty.
    ranked = engine.recommend(["鸡肉", "洋葱", "番茄"], top_k=3)
    assert ranked
    top = ranked[0].recipe.ingredients
    assert "onion" in top or "chicken" in top or "tomato" in top


def test_missing_ingredients_reported(engine):
    ranked = engine.recommend(["chicken"], top_k=3)
    assert ranked
    # The top recipe should have at least one missing non-staple ingredient.
    assert len(ranked[0].missing) >= 1
