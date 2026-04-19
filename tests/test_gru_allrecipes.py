"""Tests for the local AllRecipes parser + retriever used by the GRU path."""

from src.gru.allrecipes import ParsedRecipe, parse_recipe_blob, retrieve_recipes


def test_parse_recipe_blob_extracts_sections_and_merges_wrapped_steps():
    text = """
Chicken Pasta
Ingredients:
- 1 chicken breast
- 8 ounces pasta
Directions:
- Bring a pot of water to a boil.
- Add pasta and cook until al dente.
- Place chicken over
- medium heat until cooked through.
"""

    parsed = parse_recipe_blob(text)

    assert parsed is not None
    assert parsed.title == "Chicken Pasta"
    assert parsed.ingredient_lines == ["1 chicken breast", "8 ounces pasta"]
    assert parsed.directions == [
        "Bring a pot of water to a boil.",
        "Add pasta and cook until al dente.",
        "Place chicken over medium heat until cooked through.",
    ]


def test_retrieve_recipes_prefers_overlap_and_title_bonus():
    recipes = [
        ParsedRecipe(
            title="Chicken Pasta",
            ingredient_lines=["1 chicken breast", "8 ounces pasta"],
            directions=["Cook and serve."],
            raw_text="",
        ),
        ParsedRecipe(
            title="Tomato Soup",
            ingredient_lines=["4 tomatoes", "2 cups broth"],
            directions=["Simmer and serve."],
            raw_text="",
        ),
    ]

    matches = retrieve_recipes(recipes, "chicken, pasta", top_k=2)

    assert len(matches) == 1
    assert matches[0][1].title == "Chicken Pasta"
    assert matches[0][0] > 1.0

